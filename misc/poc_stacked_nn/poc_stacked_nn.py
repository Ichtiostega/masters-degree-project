import pandas as pd
import numpy as np
import tensorflow.keras.utils as ku
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl


def get_data(train_share: float):
    df = pd.read_csv(
        "abalone_data.csv",
        names=[
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole",
            "Shucked",
            "Viscera",
            "Shell",
            "Rings",
        ],
    )

    sexes = {"M": 0, "F": 1, "I": 2}

    df["Sex"] = df["Sex"].map(lambda x: sexes[x])

    X, Y = (
        df.drop(["Rings"], axis=1).to_numpy().astype(np.float32),
        ku.to_categorical([x // 10 for x in df["Rings"].to_numpy().astype(np.int32)]),
    )

    X_train, X_test = np.split(X, [int(X.shape[0] * train_share)])
    Y_train, Y_test = np.split(Y, [int(X.shape[0] * train_share)])

    print(*[d.shape for d in [X_train, X_test, Y_train, Y_test]])
    return X_train, X_test, Y_train, Y_test


def get_model(input_size, out_size):
    model = km.Sequential()
    model.add(kl.Dense(50, input_dim=input_size, activation="relu"))
    model.add(kl.Dense(out_size, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


X_train, X_test, Y_train, Y_test = get_data(0.8)
model = get_model(X_train.shape[1], Y_train.shape[1])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=500, verbose=1)

_, train_acc = model.evaluate(X_train, Y_train, verbose=0)
_, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print("Train: %.3f, Test: %.3f" % (train_acc, test_acc))
