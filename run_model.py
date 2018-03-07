from keras.models import load_model
import scipy.io as spio
import pickle
import sys
def main():
    model = load_model(sys.argv[1])
    x_test = spio.loadmat(sys.argv[2], squeeze_me=True)

    predicted_labels = simulateNN(model, x_test)
    pickle.dump(predicted_labels, open('predicted_labels' + '.p', 'wb'))

    # filename = 'predicted_labels.p'
    # loaded_y_pred = load_y_pred(filename)

def simulateNN(model, x_test):
    y_pred = model.predict(x_test, batch_size=128, verbose=1)
    y_pred = y_pred.argmax(1)
    return y_pred

def load_y_pred():
    y_pred = pickle.load(open(filename, "rb"))
    return y_pred

if __name__ == "__main__":
    main()
