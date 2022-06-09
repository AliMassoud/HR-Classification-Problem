from distutils import text_file
from re import X
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from pickle import dump, load


ordinal_encoder_file = "src/ML_code/Models/Ordinal_Encoder.pkl"
scaler_file = "src/ML_code/Models/Scaler.pkl"
one_hot_encoder_file = "src/ML_code/Models/One_Hot_Encoder.pkl"


def encode_target(y: pd.DataFrame):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y['Metier'].values.reshape(-1,))
    return pd.DataFrame(y, columns=['Metier'])


def get_Encoded_OneHot_Encoder(Not_Ord: pd.DataFrame) -> pd.DataFrame:
    cat_cols = Not_Ord.columns.values
    cols_encoded = []
    for col in cat_cols:
        cols_encoded += [f"{col}_{cat}" for cat in list(Not_Ord[col].unique())]

    oh_encoder = load(open(one_hot_encoder_file, "rb"))
    encoded_cols = oh_encoder.transform(Not_Ord[cat_cols])
    df_enc = pd.DataFrame(encoded_cols,
                          columns=oh_encoder.get_feature_names_out())
    dump(oh_encoder, open(one_hot_encoder_file, "wb"))
    return df_enc


def Encoded_Ordinal(categorical_features_Ord: pd.DataFrame) -> pd.DataFrame:
    enc = load(open(ordinal_encoder_file, "rb"))
    temp = enc.transform(categorical_features_Ord)
    column_names = []
    for col in categorical_features_Ord.columns:
        column_names.append(col[0])
    cat_encoded = pd.DataFrame()
    finall = pd.concat([
        cat_encoded,
        pd.DataFrame(
            temp,
            columns=column_names
        )
        ], axis=1)
    dump(enc, open(ordinal_encoder_file, "wb"))
    return finall

# split data to Numeric, Nominal and Ordinal
def split_(x: pd.DataFrame):
    new_df = (x['Technologies'].explode()
   .groupby(level=0).value_counts()
   .unstack(fill_value=0)
   .reindex(x.index, fill_value=0)
    )

    ret = x.join(new_df)
    Ordinal = ret[['Diplome', 'Ville', 'Entreprise']]
    # Not_Ordinal = x[['Technologies', 'Ville', 'Entreprise']]
    numeric = ret.drop(['Diplome', 'Technologies', 'Ville', 'Entreprise'], axis=1)
    # return Ordinal, Not_Ordinal, numeric
    return Ordinal, numeric


#  using Standard scaler on the data after I cleaned and encoding it.
def scale_final_data(x: pd.DataFrame, dataset_Type: bool) -> pd.DataFrame:
    if(dataset_Type is False):
        scaler = StandardScaler()
        t = scaler.fit(x)
        dump(scaler, open(scaler_file, "wb"))
    elif(dataset_Type is True):
        txt_file = open("data_format.txt", "r")
        columns_names = txt_file.read()
        columns_names = columns_names.split('\n')
        columns_names = [x for x in columns_names if x]
        txt_file.close()
        for i in columns_names:
            if(i not in x.columns.values):
                x[[i]] = 0
        scaler = load(open(scaler_file, "rb"))
        scaled = scaler.transform(x)
        return scaled
    return x


def before_split_data_type(dataset: pd.DataFrame):
    if('Ville,,,,' in dataset.columns.values):
        dataset['Ville'] = dataset['Ville,,,,']
        dataset.drop(['Ville,,,,'], axis=1, inplace=True)
# I am converting the Diplomes to lower case because for instance there is MSC and msc values
    for index, row in dataset.iterrows():
        dataset['Diplome'][index] = row['Diplome'].lower()
# I merged 'no Diploma' and 'no', and 'mastere' and 'master'
    for val in ['mastere']:
        dataset['Diplome'].replace(val, 'master', inplace=True)

    for val in ['no']:
        dataset['Diplome'].replace(val, 'no diploma', inplace=True)
# in the Ville column, some values have the form like "Paris,,,,", so I am removing the commas
    for index, row in dataset.iterrows():
        dataset['Ville'][index] = row['Ville'].split(",")[0]
# fill Nan values with the most repeated (mode)
    dataset = dataset.fillna(dataset.mode().iloc[0])
# extract technologies, instead of having them all in one a stirng of form C/C++ ... we have only one technology per row, (expanded the dataset).
    for index, row in dataset.iterrows():
        technologies = dataset['Technologies'][index].split('/')
        dataset['Technologies'][index] = technologies
    for val in ['NoSQ']:
        dataset.replace(val, 'NoSQL', inplace=True)
#  the values in Experience are in string format and the decimal point is represented by a cumma ',' (2,5) I am replacing them with the format (2.5)
    temp2_Experience_np = dataset['Experience'].values
    temp_experience_list = []
    for i in temp2_Experience_np:
        if(',' in i[0]):
            t = i[0].replace(',','.')
            t = float(t)
            temp_experience_list.append(t)
        else:
            t = float(i[0])
            temp_experience_list.append(t)
    Experience = np.array(temp_experience_list)
    dataset['Experience'] = Experience

    return dataset


def preprocess(Ordinal: pd.DataFrame, numeric, dataset_Typee):
    if(dataset_Typee is False):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=np.nan)
        # oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        enc.fit(Ordinal)
        # oh_encoder.fit(Not_Ordinal)
        dump(enc, open(ordinal_encoder_file, "wb"))
        # dump(oh_encoder, open(one_hot_encoder_file, "wb"))
    ordinal = Encoded_Ordinal(Ordinal)

    # Not_ordinal = get_Encoded_OneHot_Encoder(Not_Ordinal)
    ordinal.reset_index()
    numeric.reset_index()
    # Not_ordinal.reset_index()
    numeric_ordinal = numeric.join(ordinal)
    if('' in numeric_ordinal.columns.values):
        numeric_ordinal = numeric_ordinal.drop(columns='')
    if(dataset_Typee == False):
        np.savetxt('data_format.txt',numeric_ordinal.columns.values, fmt='%s')
    numeric_ordinal = numeric_ordinal.fillna(numeric_ordinal.median())
    numeric_ordinal.rename(columns = {'(Experience,)':'Experience'}, inplace = True)
    # final = numeric_norminal.join(Not_ordinal)
    numeric_ordinal.interpolate(method='linear', limit_direction='forward', inplace=True)
    numeric_ordinal.interpolate(method='linear', limit_direction='backward', inplace=True)


    x = scale_final_data(numeric_ordinal, dataset_Typee)
    return x