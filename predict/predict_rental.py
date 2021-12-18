import joblib
import pandas as pd
import numpy as np

imputer_path = "./model/rental_imputer.joblib"
rental_forest_reg_path = "./model/rental_forest_reg.joblib"
rental_lin_reg_path = "./model/rental_lin_reg.joblib"
rental_tree_reg_path = "./model/rental_tree_reg.joblib"
pipeline_path = "./model/rental_pipeline.joblib"


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

def predict_rental_value(x, imputer_path = imputer_path, model_path = rental_forest_reg_path):
  column_rental = category + numeric
  df_data_rental = x[column_rental]
  for i in category:
    df_data_rental[i] = df_data_rental[i].astype('category').cat.codes

  imputer = joblib.load(imputer_path)
  df_data_rental = df_data_rental.replace(-1, np.NaN)
  rental_imputer = imputer.transform(df_data_rental.values)
  data_rental = pd.DataFrame(rental_imputer, index=df_data_rental.index, columns=df_data_rental.columns)

  full_pipeline = joblib.load(pipeline_path)
  X_rental = full_pipeline.transform(data_rental)
  model_selection = joblib.load(model_path)
  return model_selection.predict(X_rental)