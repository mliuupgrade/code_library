# Import the model
def _load_xgb_native_booster(model_path, config_path):
    bst = Booster()
    bst.load_model(model_path)
    if config_path:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            bst.load_config(config[1])
            bst.feature_names = config[0]
    return bst


def _load_skxgb_model(model_path, bst_to_skl, config_path):
    model = XGBClassifier()
    if bst_to_skl:
        bst = _load_xgb_native_booster(model_path, config_path)
        model._Booster = bst
    else:
        model.load_model(model_path)
        if config_path:
            with open(config_path, 'rb') as f:
                feats = pickle.load(f)
                model._Booster.feature_names = feats
    return model

def load_xgb_model(model_path, model_type, convert_type=False, config_path=None):
    if model_type=='sklearn':
        if convert_type==True:
            model = _load_skxgb_model(model_path=model_path, bst_to_skl=False, config_path=config_path).get_booster()
        else:
            model = _load_skxgb_model(model_path=model_path, bst_to_skl=False, config_path=config_path)
    elif model_type=='native':
        if convert_type==True:
            model = _load_skxgb_model(model_path=model_path, bst_to_skl=True, config_path=config_path)
        else:
            model = _load_xgb_native_booster(model_path=model_path, config_path=config_path)
    else:
        print ("Please provide a valid model type ('sklearn' or 'native').")
        return
    return model