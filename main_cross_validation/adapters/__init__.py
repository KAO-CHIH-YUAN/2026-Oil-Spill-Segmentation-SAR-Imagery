# 這個檔案的目的是讓 'adapters' 資料夾成為一個 Python 套件 (package)。
# 並且，我們在這裡集中導入所有的適配器模組。
# 這樣，主程式只需要導入 'adapters' 這一個模組，
# 就可以觸發所有適配器檔案中的 @register_model 裝飾器，
# 從而自動完成所有模型的註冊。

# [Updated by Copilot]
# We commented out the explicit imports below because they create hard dependencies.
# The `training_module.py` uses `_dynamic_import_adapters()` to import these files individually
# and robustly (handling ImportErrors gracefully).
# Keeping them here causes the entire package to fail if one module (e.g. rs3mamba) is missing a dependency.

# from . import yolo_adapter
# from . import deeplab_adapter
# # from . import deeplab_adapter_aux
# from . import segformer_adapter
# from . import segformer_adapter_aux
# from . import unet_adapter
# from . import unet_adapter_aux
# from . import unetpp_adapter
# from . import paper_unet_adapter
# #from . import segnext_adapter
# from . import rs3mamba_adapter
# from . import attention_unet_adapter

# 當您新增一個新的適配器檔案，例如 'new_model_adapter.py'，
# 您只需要在這裡新增一行：
# from . import new_model_adapter
