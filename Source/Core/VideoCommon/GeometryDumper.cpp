// Copyright 2024 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/GeometryDumper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>

#include <fmt/format.h>
#include <tinygltf/tiny_gltf.h>

#include "Common/CommonPaths.h"
#include "Common/FileUtil.h"
#include "Common/Logging/Log.h"
#include "Core/ConfigManager.h"
#include "VideoCommon/AbstractTexture.h"
#include "VideoCommon/CPMemory.h"
#include "VideoCommon/TextureCacheBase.h"
#include "VideoCommon/XFMemory.h"
#include "VideoCommon/VideoEvents.h"

std::unique_ptr<GeometryDumper> g_geometry_dumper;

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

GeometryDumper::GeometryDumper()
{
  m_before_frame_hook = GetVideoEvents().before_frame_event.Register([this] { OnBeforeFrame(); });
  m_after_frame_hook =
      GetVideoEvents().after_frame_event.Register([this](Core::System& s) { OnAfterFrame(s); });
}

GeometryDumper::~GeometryDumper() = default;

void GeometryDumper::RequestCapture()
{
  m_capture_requested = true;
}

void GeometryDumper::OnBeforeFrame()
{
  if (!m_capture_requested || m_capturing)
    return;

  m_capture_requested = false;
  m_capturing = true;
  m_draws.clear();
  m_tex_to_material.clear();
  m_materials.clear();
  INFO_LOG_FMT(VIDEO, "GeometryDumper: capture started");
}

void GeometryDumper::OnAfterFrame(Core::System&)
{
  if (!m_capturing)
    return;

  m_capturing = false;
  FinalizeCapture();
}

// ---------------------------------------------------------------------------
// Vertex attribute reading helpers
// ---------------------------------------------------------------------------

float GeometryDumper::ComponentToFloat(const u8* src, ComponentFormat fmt, bool normalized)
{
  switch (fmt)
  {
  case ComponentFormat::Float:
  case ComponentFormat::InvalidFloat5:
  case ComponentFormat::InvalidFloat6:
  case ComponentFormat::InvalidFloat7:
  {
    float v;
    std::memcpy(&v, src, sizeof(float));
    return v;
  }
  case ComponentFormat::UByte:
  {
    const u8 v = *src;
    return normalized ? v / 255.0f : static_cast<float>(v);
  }
  case ComponentFormat::Byte:
  {
    s8 v;
    std::memcpy(&v, src, sizeof(s8));
    return normalized ? std::max(v / 127.0f, -1.0f) : static_cast<float>(v);
  }
  case ComponentFormat::UShort:
  {
    u16 v;
    std::memcpy(&v, src, sizeof(u16));
    return normalized ? v / 65535.0f : static_cast<float>(v);
  }
  case ComponentFormat::Short:
  {
    s16 v;
    std::memcpy(&v, src, sizeof(s16));
    return normalized ? std::max(v / 32767.0f, -1.0f) : static_cast<float>(v);
  }
  default:
    return 0.0f;
  }
}

static u32 GetComponentBytes(ComponentFormat fmt)
{
  switch (fmt)
  {
  case ComponentFormat::UByte:
  case ComponentFormat::Byte:
    return 1;
  case ComponentFormat::UShort:
  case ComponentFormat::Short:
    return 2;
  default:
    return 4;
  }
}

// ---------------------------------------------------------------------------
// CaptureDrawCall – called on GPU thread inside VertexManagerBase::Flush()
// ---------------------------------------------------------------------------

void GeometryDumper::CaptureDrawCall(const u8* vtx_data, u32 num_verts, u32 stride,
                                      const PortableVertexDeclaration& decl, const u16* idx_data,
                                      u32 num_indices,
                                      const std::array<RcTcacheEntry, 8>& tex_entries,
                                      BitSet32 used_tex)
{
  if (!vtx_data || num_verts == 0 || num_indices == 0)
    return;

  CapturedDraw dc;
  dc.num_verts = num_verts;
  dc.has_normals = decl.normals[0].enable;
  dc.has_uvs = decl.texcoords[0].enable;
  for (int ci = 0; ci < 2; ++ci)
    dc.has_colors[ci] = decl.colors[ci].enable;

  dc.positions.reserve(num_verts * 3);
  if (dc.has_normals)
    dc.normals.reserve(num_verts * 3);
  if (dc.has_uvs)
    dc.uvs.reserve(num_verts * 2);
  for (int ci = 0; ci < 2; ++ci)
    if (dc.has_colors[ci])
      dc.colors[ci].reserve(num_verts * 4);

  const AttributeFormat& pos_fmt = decl.position;
  const u32 pos_elem_bytes = GetComponentBytes(pos_fmt.type);

  // Default position matrix (used for all vertices when posmtx attr is absent).
  const u32 default_mtx_row = g_main_cp_state.matrix_index_a.PosNormalMtxIdx & 0x3f;

  for (u32 vi = 0; vi < num_verts; ++vi)
  {
    const u8* vtx = vtx_data + vi * stride;

    // Select the bone/position matrix for this vertex.
    // Per-vertex PNMTXIDX (stored as a u8 at decl.posmtx.offset) overrides the draw-call default.
    // The 6-bit value is a row index into xfmem.posMatrices; each row is 4 floats, and a 4×3
    // matrix occupies 3 consecutive rows starting at that index.
    const u32 mtx_row = decl.posmtx.enable ?
                            (static_cast<u32>(vtx[decl.posmtx.offset]) & 0x3f) :
                            default_mtx_row;
    const float* m = &xfmem.posMatrices[mtx_row * 4];

    // Position: read raw object-space value, then multiply by the 4×3 matrix.
    {
      const u8* p = vtx + pos_fmt.offset;
      const int ncomp = std::min(pos_fmt.components, 3);
      float px = 0.0f, py = 0.0f, pz = 0.0f;
      if (ncomp >= 1) px = ComponentToFloat(p + 0 * pos_elem_bytes, pos_fmt.type, false);
      if (ncomp >= 2) py = ComponentToFloat(p + 1 * pos_elem_bytes, pos_fmt.type, false);
      if (ncomp >= 3) pz = ComponentToFloat(p + 2 * pos_elem_bytes, pos_fmt.type, false);
      // result = M * [px, py, pz, 1]^T
      dc.positions.push_back(m[0] * px + m[1] * py + m[2] * pz + m[3]);
      dc.positions.push_back(m[4] * px + m[5] * py + m[6] * pz + m[7]);
      dc.positions.push_back(m[8] * px + m[9] * py + m[10] * pz + m[11]);
    }

    // Normal: transform with the 3×3 rotation sub-matrix (no translation).
    if (dc.has_normals)
    {
      const AttributeFormat& nfmt = decl.normals[0];
      const u8* p = vtx + nfmt.offset;
      const u32 ne = GetComponentBytes(nfmt.type);
      const float nx = ComponentToFloat(p + 0 * ne, nfmt.type, true);
      const float ny = ComponentToFloat(p + 1 * ne, nfmt.type, true);
      const float nz = ComponentToFloat(p + 2 * ne, nfmt.type, true);
      dc.normals.push_back(m[0] * nx + m[1] * ny + m[2] * nz);
      dc.normals.push_back(m[4] * nx + m[5] * ny + m[6] * nz);
      dc.normals.push_back(m[8] * nx + m[9] * ny + m[10] * nz);
    }

    // UV0 – GX uses top-left UV origin, identical to glTF's convention, so no V-flip is needed
    // here. Blender's glTF importer applies its own V-flip (glTF top-left → Blender bottom-left);
    // adding a second flip here would double-invert and produce upside-down textures.
    if (dc.has_uvs)
    {
      const AttributeFormat& ufmt = decl.texcoords[0];
      const u8* p = vtx + ufmt.offset;
      const u32 ue = GetComponentBytes(ufmt.type);
      dc.uvs.push_back(ComponentToFloat(p + 0 * ue, ufmt.type, false));
      dc.uvs.push_back(ComponentToFloat(p + 1 * ue, ufmt.type, false));
    }

    // Color channels – GX supports 2 (COLOR0/COLOR1); capture both if present.
    for (int ci = 0; ci < 2; ++ci)
    {
      if (!dc.has_colors[ci])
        continue;
      const AttributeFormat& cfmt = decl.colors[ci];
      const u8* p = vtx + cfmt.offset;
      const u32 ce = GetComponentBytes(cfmt.type);
      for (int c = 0; c < 4; ++c)
      {
        const float cf = ComponentToFloat(p + c * ce, cfmt.type, false);
        dc.colors[ci].push_back(static_cast<u8>(std::clamp(cf, 0.0f, 255.0f)));
      }
    }
  }

  // IndexGenerator may emit 0xFFFF primitive-restart markers when the backend supports it
  // (g_backend_info.bSupportsPrimitiveRestart).  In that mode the index buffer is in
  // TRIANGLE_STRIP layout with 0xFFFF used as a segment separator, not as a real vertex index.
  // We detect which mode is active by checking whether any 0xFFFF values are present, then
  // convert to a plain triangle list.
  //
  // When no restart markers are present the buffer is already a flat triangle list — but only
  // copy it when num_indices is divisible by 3; non-divisible counts mean line/point draw calls
  // which we intentionally skip.
  {
    constexpr u16 kRestart = 0xFFFF;
    const bool has_restart =
        std::any_of(idx_data, idx_data + num_indices, [](u16 v) { return v == kRestart; });
    if (has_restart)
    {
      // Split at restart markers; each run is a triangle strip — convert to triangles.
      size_t seg_start = 0;
      for (size_t k = 0; k <= num_indices; ++k)
      {
        const bool end = (k == num_indices) || (idx_data[k] == kRestart);
        if (end)
        {
          const size_t seg_len = k - seg_start;
          for (size_t j = 2; j < seg_len; ++j)
          {
            const u16 v0 = idx_data[seg_start + j - 2];
            const u16 v1 = idx_data[seg_start + j - 1];
            const u16 v2 = idx_data[seg_start + j];
            if (v0 == v1 || v1 == v2 || v0 == v2)
              continue;  // degenerate — skip
            if (j % 2 == 0)  // even triangle: CCW as-is
            {
              dc.indices.push_back(v0);
              dc.indices.push_back(v1);
              dc.indices.push_back(v2);
            }
            else  // odd triangle: swap first two to preserve winding
            {
              dc.indices.push_back(v1);
              dc.indices.push_back(v0);
              dc.indices.push_back(v2);
            }
          }
          seg_start = k + 1;
        }
      }
    }
    else if (num_indices % 3 == 0)
    {
      // Flat triangle list (no primitive restart) — copy verbatim.
      dc.indices.assign(idx_data, idx_data + num_indices);
    }
    // else: line or point draw call (num_indices % 3 != 0) — intentionally skipped.
  }

  // Use the first active texture slot as the material texture.
  // Copy the shared_ptr so the TCacheEntry stays alive across the frame boundary.
  for (const u32 i : used_tex)
  {
    if (tex_entries[i])
    {
      dc.tex_hash = tex_entries[i]->hash;
      dc.tex_entry = tex_entries[i];
      break;
    }
  }

  m_draws.push_back(std::move(dc));
}

// ---------------------------------------------------------------------------
// FinalizeCapture – called on GPU thread at end of frame
// ---------------------------------------------------------------------------

static std::string GenerateOutputDir()
{
  const std::string game_id = SConfig::GetInstance().GetGameID();

  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  struct tm local_tm
  {
  };
#ifdef _WIN32
  localtime_s(&local_tm, &t);
#else
  localtime_r(&t, &local_tm);
#endif
  char ts[32];
  std::strftime(ts, sizeof(ts), "%Y-%m-%d_%H-%M-%S", &local_tm);

  // ScreenShots/<GameID>/3D/<GameID>_<timestamp>/
  const std::string dir =
      fmt::format("{}{}{}3D{}{}_{}", File::GetUserPath(D_SCREENSHOTS_IDX), game_id, DIR_SEP,
                  DIR_SEP, game_id, ts);
  return dir;
}

void GeometryDumper::FinalizeCapture()
{
  if (m_draws.empty())
  {
    INFO_LOG_FMT(VIDEO, "GeometryDumper: no geometry captured this frame");
    return;
  }

  const std::string output_dir = GenerateOutputDir();
  if (!File::CreateFullPath(output_dir + DIR_SEP))
  {
    ERROR_LOG_FMT(VIDEO, "GeometryDumper: failed to create output directory '{}'", output_dir);
    return;
  }

  INFO_LOG_FMT(VIDEO, "GeometryDumper: saving {} draw calls to '{}'", m_draws.size(), output_dir);
  WriteGltf(output_dir);
}

// ---------------------------------------------------------------------------
// Material / texture helpers
// ---------------------------------------------------------------------------

int GeometryDumper::GetOrCreateMaterial(const std::string& output_dir, u64 tex_hash,
                                         const RcTcacheEntry& entry)
{
  auto it = m_tex_to_material.find(tex_hash);
  if (it != m_tex_to_material.end())
    return it->second;

  const int mat_idx = static_cast<int>(m_materials.size());
  MaterialEntry mat;

  if (entry && entry->texture && tex_hash != 0)
  {
    const AbstractTextureFormat fmt = entry->texture->GetConfig().format;
    const bool can_save = !AbstractTexture::IsCompressedFormat(fmt) &&
                          !AbstractTexture::IsDepthFormat(fmt) &&
                          fmt != AbstractTextureFormat::RGBA16F;
    if (can_save)
    {
      mat.tex_filename = fmt::format("tex_{:016x}.png", tex_hash);
      const std::string full_path = output_dir + DIR_SEP + mat.tex_filename;
      if (!File::Exists(full_path))
      {
        if (!entry->texture->Save(full_path, 0))
          WARN_LOG_FMT(VIDEO, "GeometryDumper: failed to save texture '{}'", full_path);
      }
    }
  }

  m_tex_to_material[tex_hash] = mat_idx;
  m_materials.push_back(std::move(mat));
  return mat_idx;
}

// ---------------------------------------------------------------------------
// glTF writer
// ---------------------------------------------------------------------------

// Append raw bytes to buf and return the starting offset.
static size_t AppendBytes(std::vector<unsigned char>& buf, const void* data, size_t len)
{
  const size_t offset = buf.size();
  const auto* bytes = reinterpret_cast<const unsigned char*>(data);
  buf.insert(buf.end(), bytes, bytes + len);
  return offset;
}

// Pad buf to the given alignment with zero bytes.
static void AlignBuffer(std::vector<unsigned char>& buf, size_t alignment)
{
  while (buf.size() % alignment != 0)
    buf.push_back(0);
}

void GeometryDumper::WriteGltf(const std::string& output_dir)
{
  tinygltf::Model model;
  model.asset.version = "2.0";
  model.asset.generator = "Dolphin Emulator 3D Screenshot";

  tinygltf::Scene scene;
  scene.name = "Scene";
  model.defaultScene = 0;
  model.scenes.push_back(std::move(scene));

  // One default sampler: linear filtering, repeat wrap.
  {
    tinygltf::Sampler sampler;
    sampler.magFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
    sampler.minFilter = TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR;
    sampler.wrapS = TINYGLTF_TEXTURE_WRAP_REPEAT;
    sampler.wrapT = TINYGLTF_TEXTURE_WRAP_REPEAT;
    model.samplers.push_back(std::move(sampler));
  }

  // Single shared binary buffer accumulating all mesh data.
  std::vector<unsigned char> bin;

  // Helper: get-or-create a tinygltf texture index from an image filename.
  // tinygltf images are deduplicated by URI.
  auto GetOrCreateGltfTexture = [&](const std::string& tex_filename) -> int {
    for (int i = 0; i < static_cast<int>(model.images.size()); ++i)
    {
      if (model.images[i].uri == tex_filename)
      {
        for (int j = 0; j < static_cast<int>(model.textures.size()); ++j)
        {
          if (model.textures[j].source == i)
            return j;
        }
      }
    }
    tinygltf::Image img;
    img.uri = tex_filename;
    const int img_idx = static_cast<int>(model.images.size());
    model.images.push_back(std::move(img));

    tinygltf::Texture tex;
    tex.source = img_idx;
    tex.sampler = 0;
    const int tex_idx = static_cast<int>(model.textures.size());
    model.textures.push_back(std::move(tex));
    return tex_idx;
  };

  // Process draw calls.
  for (int di = 0; di < static_cast<int>(m_draws.size()); ++di)
  {
    const CapturedDraw& dc = m_draws[di];
    if (dc.indices.empty() || dc.positions.empty())
      continue;

    tinygltf::Mesh mesh;
    mesh.name = fmt::format("draw_{}", di);
    tinygltf::Primitive prim;
    prim.mode = TINYGLTF_MODE_TRIANGLES;

    // Assign material (also saves texture PNG on first use).
    prim.material = GetOrCreateMaterial(output_dir, dc.tex_hash, dc.tex_entry);

    // Helper lambda: add a bufferView + accessor and return the accessor index.
    auto AddAccessor = [&](const void* data, size_t byte_len, size_t align, int component_type,
                           int type, u32 count, bool normalized = false) -> int {
      AlignBuffer(bin, align);
      const size_t bv_offset = AppendBytes(bin, data, byte_len);

      tinygltf::BufferView bv;
      bv.buffer = 0;
      bv.byteOffset = bv_offset;
      bv.byteLength = byte_len;
      // Position/attribute data vs index data
      bv.target = (component_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT &&
                   type == TINYGLTF_TYPE_SCALAR) ?
                      TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER :
                      TINYGLTF_TARGET_ARRAY_BUFFER;
      const int bv_idx = static_cast<int>(model.bufferViews.size());
      model.bufferViews.push_back(std::move(bv));

      tinygltf::Accessor acc;
      acc.bufferView = bv_idx;
      acc.byteOffset = 0;
      acc.count = count;
      acc.componentType = component_type;
      acc.type = type;
      acc.normalized = normalized;
      const int acc_idx = static_cast<int>(model.accessors.size());
      model.accessors.push_back(std::move(acc));
      return acc_idx;
    };

    // ---- POSITION (required, include min/max per spec) ----
    {
      float xmin = std::numeric_limits<float>::max();
      float ymin = xmin, zmin = xmin;
      float xmax = std::numeric_limits<float>::lowest();
      float ymax = xmax, zmax = xmax;
      for (u32 vi = 0; vi < dc.num_verts; ++vi)
      {
        xmin = std::min(xmin, dc.positions[vi * 3 + 0]);
        ymin = std::min(ymin, dc.positions[vi * 3 + 1]);
        zmin = std::min(zmin, dc.positions[vi * 3 + 2]);
        xmax = std::max(xmax, dc.positions[vi * 3 + 0]);
        ymax = std::max(ymax, dc.positions[vi * 3 + 1]);
        zmax = std::max(zmax, dc.positions[vi * 3 + 2]);
      }

      AlignBuffer(bin, 4);
      const size_t bv_offset =
          AppendBytes(bin, dc.positions.data(), dc.positions.size() * sizeof(float));

      tinygltf::BufferView bv;
      bv.buffer = 0;
      bv.byteOffset = bv_offset;
      bv.byteLength = dc.positions.size() * sizeof(float);
      bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
      const int bv_idx = static_cast<int>(model.bufferViews.size());
      model.bufferViews.push_back(std::move(bv));

      tinygltf::Accessor acc;
      acc.bufferView = bv_idx;
      acc.byteOffset = 0;
      acc.count = dc.num_verts;
      acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
      acc.type = TINYGLTF_TYPE_VEC3;
      acc.minValues = {static_cast<double>(xmin), static_cast<double>(ymin),
                       static_cast<double>(zmin)};
      acc.maxValues = {static_cast<double>(xmax), static_cast<double>(ymax),
                       static_cast<double>(zmax)};
      prim.attributes["POSITION"] = static_cast<int>(model.accessors.size());
      model.accessors.push_back(std::move(acc));
    }

    // ---- NORMAL ----
    if (dc.has_normals && !dc.normals.empty())
    {
      const int acc_idx =
          AddAccessor(dc.normals.data(), dc.normals.size() * sizeof(float), 4,
                      TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, dc.num_verts);
      prim.attributes["NORMAL"] = acc_idx;
    }

    // ---- TEXCOORD_0 ----
    if (dc.has_uvs && !dc.uvs.empty())
    {
      const int acc_idx =
          AddAccessor(dc.uvs.data(), dc.uvs.size() * sizeof(float), 4,
                      TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, dc.num_verts);
      prim.attributes["TEXCOORD_0"] = acc_idx;
    }

    // ---- COLOR_0 / COLOR_1 (normalized u8 RGBA → [0,1]) ----
    for (int ci = 0; ci < 2; ++ci)
    {
      if (!dc.has_colors[ci] || dc.colors[ci].empty())
        continue;
      const int acc_idx =
          AddAccessor(dc.colors[ci].data(), dc.colors[ci].size() * sizeof(u8), 1,
                      TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, TINYGLTF_TYPE_VEC4, dc.num_verts,
                      /*normalized=*/true);
      prim.attributes[fmt::format("COLOR_{}", ci)] = acc_idx;
    }

    // ---- INDICES (u16) ----
    {
      AlignBuffer(bin, 2);
      const size_t bv_offset =
          AppendBytes(bin, dc.indices.data(), dc.indices.size() * sizeof(u16));

      tinygltf::BufferView bv;
      bv.buffer = 0;
      bv.byteOffset = bv_offset;
      bv.byteLength = dc.indices.size() * sizeof(u16);
      bv.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
      const int bv_idx = static_cast<int>(model.bufferViews.size());
      model.bufferViews.push_back(std::move(bv));

      tinygltf::Accessor acc;
      acc.bufferView = bv_idx;
      acc.byteOffset = 0;
      acc.count = static_cast<u32>(dc.indices.size());
      acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
      acc.type = TINYGLTF_TYPE_SCALAR;
      prim.indices = static_cast<int>(model.accessors.size());
      model.accessors.push_back(std::move(acc));
    }

    mesh.primitives.push_back(std::move(prim));

    const int mesh_idx = static_cast<int>(model.meshes.size());
    model.meshes.push_back(std::move(mesh));

    tinygltf::Node node;
    node.mesh = mesh_idx;
    // Positions were already transformed to eye-space per-vertex during capture;
    // no additional node transform is needed.

    const int node_idx = static_cast<int>(model.nodes.size());
    model.nodes.push_back(std::move(node));
    model.scenes[0].nodes.push_back(node_idx);
  }

  // Build glTF materials after all GetOrCreateMaterial calls are done.
  for (const MaterialEntry& mat : m_materials)
  {
    tinygltf::Material gmat;
    gmat.name = mat.tex_filename.empty() ? "untextured" : mat.tex_filename;
    gmat.doubleSided = true;

    if (!mat.tex_filename.empty())
    {
      const int tex_idx = GetOrCreateGltfTexture(mat.tex_filename);
      gmat.pbrMetallicRoughness.baseColorTexture.index = tex_idx;
      gmat.pbrMetallicRoughness.baseColorTexture.texCoord = 0;
    }
    gmat.pbrMetallicRoughness.metallicFactor = 0.0;
    gmat.pbrMetallicRoughness.roughnessFactor = 1.0;
    model.materials.push_back(std::move(gmat));
  }

  // Finalize binary buffer.
  tinygltf::Buffer buf;
  buf.name = "meshdata";
  buf.uri = "scene.bin";
  buf.data = std::move(bin);
  model.buffers.push_back(std::move(buf));

  const std::string gltf_path = output_dir + DIR_SEP + "scene.gltf";
  tinygltf::TinyGLTF writer;
  if (!writer.WriteGltfSceneToFile(&model, gltf_path,
                                   /*embedImages=*/false,
                                   /*embedBuffers=*/false,
                                   /*prettyPrint=*/true,
                                   /*writeBinary=*/false))
  {
    ERROR_LOG_FMT(VIDEO, "GeometryDumper: failed to write '{}'", gltf_path);
    return;
  }

  INFO_LOG_FMT(VIDEO, "GeometryDumper: 3D screenshot saved to '{}'", gltf_path);
}
