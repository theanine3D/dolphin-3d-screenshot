// Copyright 2024 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Common/BitSet.h"
#include "Common/CommonTypes.h"
#include "Common/HookableEvent.h"
#include "VideoCommon/NativeVertexFormat.h"
#include "VideoCommon/TextureCacheBase.h"

namespace Core
{
class System;
}

// Captures one frame of 3D geometry (vertices, indices, textures) and writes a glTF 2.0 file.
//
// Usage: call RequestCapture() once from any thread. At the start of the next GPU frame the
// dumper begins recording every draw call. At end-of-frame it writes the glTF and resets.
class GeometryDumper
{
public:
  GeometryDumper();
  ~GeometryDumper();

  // Thread-safe: request a 3D screenshot on the next frame.
  void RequestCapture();

  // Called from the GPU thread during VertexManagerBase::Flush(), before CommitBuffer().
  // vtx_data    – pointer to raw vertex data (CPU-accessible streaming buffer)
  // num_verts   – number of unique vertices
  // stride      – byte stride per vertex
  // decl        – attribute offsets and formats
  // idx_data    – pointer to index data (u16 values)
  // num_indices – number of indices
  // tex_entries – TCacheEntry* per texture unit (nullptr if unit unused)
  // used_tex    – bit mask of active texture units
  void CaptureDrawCall(const u8* vtx_data, u32 num_verts, u32 stride,
                       const PortableVertexDeclaration& decl, const u16* idx_data,
                       u32 num_indices,
                       const std::array<RcTcacheEntry, 8>& tex_entries,
                       BitSet32 used_tex);

  bool IsCapturing() const { return m_capturing; }

private:
  struct CapturedDraw
  {
    std::vector<float> positions;  // float3 per vertex
    std::vector<float> normals;    // float3 per vertex (may be empty)
    std::vector<float> uvs;        // float2 per vertex, first UV set (may be empty)
    std::vector<u8> colors;        // u8 RGBA per vertex (may be empty)
    std::vector<u16> indices;
    u32 num_verts = 0;
    bool has_normals = false;
    bool has_uvs = false;
    bool has_colors = false;
    // Identifies the texture used. 0 means no texture.
    u64 tex_hash = 0;
    // Shared ownership keeps the TCacheEntry alive across the frame boundary.
    RcTcacheEntry tex_entry;
  };

  void OnBeforeFrame();
  void OnAfterFrame(Core::System& system);
  void FinalizeCapture();
  void WriteGltf(const std::string& output_dir);

  // Returns the index into m_materials (creating a new entry if needed).
  // Saves the texture PNG to output_dir at the same time.
  int GetOrCreateMaterial(const std::string& output_dir, u64 tex_hash,
                          const RcTcacheEntry& entry);

  // Converts a raw vertex attribute value to float, normalising integers.
  static float ComponentToFloat(const u8* src, ComponentFormat fmt, bool normalized);

  bool m_capture_requested = false;
  bool m_capturing = false;

  std::vector<CapturedDraw> m_draws;

  // tex_hash → material index in m_materials
  std::unordered_map<u64, int> m_tex_to_material;

  struct MaterialEntry
  {
    std::string tex_filename;  // basename of PNG, empty = no texture
  };
  std::vector<MaterialEntry> m_materials;

  Common::EventHook m_before_frame_hook;
  Common::EventHook m_after_frame_hook;
};

extern std::unique_ptr<GeometryDumper> g_geometry_dumper;
