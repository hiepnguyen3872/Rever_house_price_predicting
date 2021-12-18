import joblib
import pandas as pd
import numpy as np

imputer_path = "./model/sell_imputer.joblib"
sell_forest_reg_path = "./model/sell_forest_reg.joblib"
sell_lin_reg_path = "./model/sell_lin_reg.joblib"
sell_tree_reg_path = "./model/sell_tree_reg.joblib"
pipeline_path = "./model/sell_pipeline.joblib"


category = ['furniture_status',
     'property_type', 'direction', 'ownership',
     'has_3d', 'service_type',
     'balcony_direction', 'content_status',
     'architectural_style', 'exclusive',
     'pool', 'open24h', 'garage', 'sauna_bath',
     'working_space', 'relax_room', 'elevator', 'gym', 'cable', 'internet', 'pet', 'steam_bath',
     'smart_home', 'tv', 'fridge', 'store_house', 'smart_drying_rig',
     'gas_stove', 'mini_bar', 'microwave', 'helper_room',
     'washing_machine', 'oven', 'fire_detection', 'water_heater',
     'password_lock', 'kitchen_hood', 'dryer', 'sound_equipment',
     'air_conditioner', 'fingerprint_lock', 'security_camera', 'garden',
     'magnetic_card_lock', 'city', 'floor_number',
     'neighborhood_id', 'district_id', 'street_id', 'city_id', 'ward_id',
     'kitchen_cabinet', 'bed', 'sofa',  'dining_table', 'balcony',
     'kitchen_equipment', 'multimedia',
     'makeup_table', 'wardrobe', 'kitchen', 'table',
     'pillow_cushions', 'shoe_cabinet', 'washstand',
     'kitchen_island', 'bathtub',
     'bedside_cupboard', 'decorative_fight',
     'wet_kitchen', 'tv_shelf',
     'bookshelf', 'wall_cabinet', 'ceiling_light', 'toilet_bowl',
     'wood_floor', 'kitchen_cabinet_above', 'table_lamp', 'dry_kitchen', 'tv_cabinet', 'liquor_cabinet']

numeric = ['num_bath_room', 'num_bed_room', 'area']

def predict_sell_value(x, imputer_path = imputer_path, model_path = sell_forest_reg_path):
  column_sell = category + numeric
  df_data_sell = x[column_sell]
  for i in category:
    df_data_sell[i] = df_data_sell[i].astype('category').cat.codes

  imputer = joblib.load(imputer_path)
  df_data_sell = df_data_sell.replace(-1, np.NaN)
  sell_imputer = imputer.transform(df_data_sell.values)
  data_sell = pd.DataFrame(sell_imputer, index=df_data_sell.index, columns=df_data_sell.columns)

  full_pipeline = joblib.load(pipeline_path)
  X_sell = full_pipeline.transform(data_sell)
  model_selection = joblib.load(model_path)
  return model_selection.predict(X_sell)*0.9