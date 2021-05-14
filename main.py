import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing


def read_dataset():
    data = pd.read_csv('assets/dataset.csv')
    data.InitialPCRDiagnosis = data.InitialPCRDiagnosis.fillna("NA")
    data.Age = data.Age.fillna(0)
    data.Sex = data.Sex.fillna("NA")
    data.Region = data.Region.fillna("NA")
    data.CommunityTransmission = data.CommunityTransmission.fillna("NA")
    data.NumberOfFamilyMembersInfected = data.NumberOfFamilyMembersInfected.fillna(0)
    data.neutrophil = data.neutrophil.fillna(0)
    data.neutrophilCategorical = data.neutrophilCategorical.fillna("NA")
    data.serumLevelsOfWhiteBloodCell = data.serumLevelsOfWhiteBloodCell.fillna(0)
    data.serumLevelsOfWhiteBloodCellCategorical = data.serumLevelsOfWhiteBloodCellCategorical.fillna("NA")
    data.lymphocytes = data.lymphocytes.fillna("NA")
    data.lymphocytes = data.lymphocytes.astype(str)
    data.lymphocytesCategorical = data.lymphocytesCategorical.fillna("NA")
    data.Plateletes = data.Plateletes.fillna(0)
    data.CReactiveProteinLevels = data.CReactiveProteinLevels.fillna("NA")
    data.CReactiveProteinLevels = data.CReactiveProteinLevels.astype(str)
    data.CReactiveProteinLevelsCategorical = data.CReactiveProteinLevelsCategorical.fillna("NA")
    data.Eosinophils = data.Eosinophils.fillna(0)
    data.Redbloodcells = data.Redbloodcells.fillna(0)
    data.Hemoglobin = data.Hemoglobin.fillna(0)
    data.Procalcitonin = data.Procalcitonin.fillna(0)
    data.DurationOfIllness = data.DurationOfIllness.fillna(0)
    data.DaysToDeath = data.DaysToDeath.fillna(0)
    data.DaysInIncubation = data.DaysInIncubation.fillna(0)
    data.CTscanResults = data.CTscanResults.fillna("NA")
    data.XrayResults = data.XrayResults.fillna("NA")
    data.RiskFactors = data.RiskFactors.fillna("NA")
    data.SmokingStatus = data.SmokingStatus.fillna("NA")
    data.VapingStatus = data.VapingStatus.fillna("NA")
    data.NumberAffectedLobes = data.NumberAffectedLobes.fillna(0)
    data.GroundGlassOpacity = data.GroundGlassOpacity.fillna("NA")
    data.Asymptomatic = data.Asymptomatic.fillna("NA")
    data.Diarrhea = data.Diarrhea.fillna("NA")
    data.Fever = data.Fever.fillna("NA")
    data.Coughing = data.Coughing.fillna("NA")
    data.ShortnessOfBreath = data.ShortnessOfBreath.fillna("NA")
    data.SoreThroat = data.SoreThroat.fillna("NA")
    data.NauseaVomitting = data.NauseaVomitting.fillna("NA")
    data.TimeBetweenAdmissionAndDiagnosis = data.TimeBetweenAdmissionAndDiagnosis.fillna(0)
    data.Pregnant = data.Pregnant.fillna("NA")
    data.BabyDeath = data.BabyDeath.fillna("NA")
    data.PrematureDelivery = data.PrematureDelivery.fillna("NA")
    data.Hematocrit = data.Hematocrit.fillna(0)
    data.Temperature = data.Temperature.fillna(0)
    data.ActivatedPartialThromboplastinTime = data.ActivatedPartialThromboplastinTime.fillna(0)
    data.Fibrinogen = data.Fibrinogen.fillna(0)
    data.Urea = data.Urea.fillna(0)
    data.Fatigue = data.Fatigue.fillna("NA")
    data.Monocytes = data.Monocytes.fillna(0)
    data.Basophil = data.Basophil.fillna(0)
    data.Cancer = data.Cancer.fillna("NA")
    data.Thrombocytes = data.Thrombocytes.fillna(0)

    data.Diagnosis = data.Diagnosis.map({'H1N1': 0, 'COVID19': 1})

    data.to_csv("output/dataset.csv")

    return data
    # print(data.head)


def split(data):
    train, test = train_test_split(data, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    return train, test, val


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Diagnosis')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_normalization_layer(name, dataset):
    normalizer = preprocessing.Normalization()
    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)
    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)

    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())
    feature_ds = feature_ds.map(index)
    encoder.adapt(feature_ds)

    return lambda feature: encoder(index(feature))


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.savefig(f"output/{string}.png", bbox_inches='tight')
    plt.show()


def config(train, test, val):
    batch_size = 64
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    all_inputs = []
    encoded_features = []
    header_cols = ['neutrophil', 'Age', 'NumberOfFamilyMembersInfected', 'serumLevelsOfWhiteBloodCell', 'Plateletes',
                   'Eosinophils', 'Redbloodcells', 'Hemoglobin', 'Procalcitonin', 'DurationOfIllness',
                   'DaysToDeath', 'DaysInIncubation', 'NumberAffectedLobes', 'TimeBetweenAdmissionAndDiagnosis',
                   'Hematocrit',
                   'Temperature', 'ActivatedPartialThromboplastinTime', 'Fibrinogen', 'Urea', 'Monocytes', 'Basophil',
                   'Thrombocytes']
    for header in header_cols:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    categorical_cols = ['CReactiveProteinLevels', 'lymphocytes', 'InitialPCRDiagnosis', 'Sex', 'Region',
                        'CommunityTransmission',
                        'neutrophilCategorical', 'serumLevelsOfWhiteBloodCellCategorical', 'lymphocytesCategorical',
                        'CReactiveProteinLevelsCategorical', 'CTscanResults', 'XrayResults', 'RiskFactors',
                        'SmokingStatus', 'VapingStatus', 'GroundGlassOpacity', 'Asymptomatic', 'Diarrhea', 'Fever',
                        'Coughing', 'ShortnessOfBreath', 'SoreThroat', 'NauseaVomitting', 'Pregnant', 'BabyDeath',
                        'PrematureDelivery', 'Fatigue', 'Cancer']
    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                     max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(256, activation="relu")(all_features)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR", to_file="output/plot_model.png")

    history = model.fit(train_ds, epochs=100, validation_data=val_ds)
    # model.save("output/model.h5")
    loss, accuracy = model.evaluate(test_ds)

    print("Accuracy", accuracy)
    print("Loss", loss)

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


if __name__ == '__main__':
    data = read_dataset()
    train, test, val = split(data)
    config(train, test, val)


