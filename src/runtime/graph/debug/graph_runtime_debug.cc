/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */
#include <sys/time.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include <dmlc/memory_io.h>
#include <dmlc/json.h>
#include <numeric>
#include "../graph_runtime.h"
#include <cmath>

namespace tvm {
namespace runtime {

#define CHECK_NONE 0x0
#define CHECK_NAN 0x1
#define CHECK_INF 0x2

/*!
 * \brief Graph runtime with debug .
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
class GraphRuntimeDebug : public GraphRuntime {

  int DebugRun(int index) {
    struct timeval tp;
    int64_t start_time = 0;
    int64_t end_time = 0;
    if (op_execs()[index]) {
      gettimeofday(&tp, NULL);
      start_time = int64_t(tp.tv_sec * 1000000L + tp.tv_usec);
      op_execs()[index]();
      gettimeofday(&tp, NULL);
      end_time = int64_t(tp.tv_sec * 1000000L + tp.tv_usec);
      for (size_t j = 0; j < NumOutputs(index); j++) {
          uint32_t eid = GetEntryId(index, j);
          TVM_CCALL(TVMArrayCopyFromTo(&data_entry()[eid], debug_buffers_[eid], nullptr));
      }
    }
    return end_time - start_time;
  }

  /*!
   * \brief Check whether the data contains NAN or INF.
   * \param data The data pointer.
   * \param check_flag The flag which denotes whether to check NAN or INF.
   */
  void CheckNanOrInf(DLTensor* data, int check_flag) {
    if (check_flag == CHECK_NONE) {
        return;
    }
    size_t size = 1;
    for (tvm_index_t i = 0; i < data->ndim; ++i) {
       size *= data->shape[i];
    }
    size *= (data->dtype.bits * data->dtype.lanes + 7) / 8;
    for (size_t i=0; (i < size); ++i) {
        if ((check_flag && CHECK_NAN) && std::isnan(((float *)data->data)[i])) {
            printf("\nERROR: NAN FOUND at index=%ld, val=%f", i, ((float *)data->data)[i]);
            break;
        }
        if ((check_flag && CHECK_INF) && std::isinf(((float *)data->data)[i])) {
            printf("\nERROR: INF FOUND at index=%ld, val=%f", i, ((float *)data->data)[i]);
            break;
        }
    }
  }

  /*!
   * \brief Set the debug buffer to copy the output of each operation.
   * \param data The data pointer.
   */
  void SetDebugBuffer(DLTensor* data) {
      debug_buffers_.push_back(data);
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self);

  private:
  /*! \brief debug buffer storage pool */
  std::vector<DLTensor*> debug_buffers_;
};


PackedFunc GraphRuntimeDebug::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "set_debug_buffer") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->SetDebugBuffer(args[0]);
      });
  } else if (name == "debug_run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->DebugRun(args[0]);
      });
  } else {
     return GraphRuntime::GetFunction(name, sptr_to_self);
  }
}

Module DebugGraphRuntimeCreate(std::string sym_json,
                          tvm::runtime::Module m,
                          int device_type,
                          int device_id) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id   = device_id;
  std::shared_ptr<GraphRuntimeDebug> exec = std::make_shared<GraphRuntimeDebug>();
  exec->Init(sym_json, m, ctx);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_debug.create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = DebugGraphRuntimeCreate(args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_GLOBAL("tvm.graph_runtime_debug.remote_create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* mhandle = args[1];
    *rv = DebugGraphRuntimeCreate(args[0],
                             *static_cast<tvm::runtime::Module*>(mhandle),
                             args[2], args[3]);
  });
}  // namespace runtime
}  // namespace tvm
