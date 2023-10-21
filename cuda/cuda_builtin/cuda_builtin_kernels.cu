// built-in update kernel for BindlessArray
struct alignas(16u) BindlessSlot {
    unsigned long long buffer;
    unsigned long long buffer_size;
    unsigned long long tex2d;
    unsigned long long tex3d;
};

static_assert(sizeof(BindlessSlot) == 32u, "");

struct alignas(16) SlotModification {
    struct Buffer {
        unsigned long long handle;
        unsigned long long size;
        unsigned int op;
    };
    struct Texture {
        unsigned long long handle;
        unsigned int sampler;// not used; processed on host
        unsigned int op;
    };
    unsigned long long slot;
    Buffer buffer;
    Texture tex2d;
    Texture tex3d;
};

static_assert(sizeof(SlotModification) == 64u, "");

extern "C" __global__ void update_bindless_array(BindlessSlot *__restrict__ array,
                                                 const SlotModification *__restrict__ mods,
                                                 unsigned int n) {
    constexpr auto op_update = 1u;
    constexpr auto op_remove = 2u;
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) [[likely]] {
        auto m = mods[tid];
        auto slot_id = m.slot;
        auto slot = array[slot_id];
        if (m.buffer.op == op_update) {
            slot.buffer = m.buffer.handle;
            slot.buffer_size = m.buffer.size;
        } else if (m.buffer.op == op_remove) {
            slot.buffer = 0u;
            slot.buffer_size = 0u;
        }
        if (m.tex2d.op == op_update) {
            slot.tex2d = m.tex2d.handle;
        } else if (m.tex2d.op == op_remove) {
            slot.tex2d = 0u;
        }
        if (m.tex3d.op == op_update) {
            slot.tex3d = m.tex3d.handle;
        } else if (m.tex3d.op == op_remove) {
            slot.tex3d = 0u;
        }
        array[slot_id] = slot;
    }
}
