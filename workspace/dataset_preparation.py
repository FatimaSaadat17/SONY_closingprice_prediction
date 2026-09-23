import numpy as np

def split_data(data):
    # define the train and test proportions of the dataset = train (70%) and test (15%) and val (15%)

    # sort the data by index date before splitting
    data = data.sort_index()

    # define the training size and test size
    train_size = int(np.round(len(data) * .70))
    dates_test = data.index[train_size:]
    val_size = int(np.round(len(data) * .15))

    # define X features and y target columns
    X = data.loc[:, [c for c in data.columns if c != 'y']]
    y = data['y']

    # split the data set accordingly
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]
    X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]


    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)} | Val size {len(X_val)}")
    print(f"Train period: {data.index[0].date()} to {data.index[train_size-1].date()}")
    print(f"Test + val period : {data.index[train_size].date()} to {data.index[-1].date()}")

    return X_train, y_train, X_val, y_val, X_test, y_test



def create_sequences_np(X_data, y_data, seq_length):
    xs, ys = [], []
    
    # Iterate over the rows of the array
    for i in range(len(X_data) - seq_length):
        # Slice rows from i to i+seq_length, and pick column index 1
        x = X_data[i : i + seq_length]
        # Pick the row at i+seq_length, and pick column index 1
        y = y_data[i + seq_length]
        
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)



