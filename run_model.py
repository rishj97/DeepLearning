from keras.models import load_model
import scipy.io as spio
import pickle
import sys
import multiprocessing
import numpy as np

def main():
    model = load_model(sys.argv[1])
    x_test = spio.loadmat(sys.argv[2], squeeze_me=True)

    data = spio.loadmat(sys.argv[2], squeeze_me=True)
    x_test = data['datasetInputs'][2]
    x_test = normalize_input(x_test)

    predicted_labels = simulateNN(model, x_test)
    pickle.dump(predicted_labels, open('predicted_labels' + '.p', 'wb'))

    # filename = 'predicted_labels.p'
    # loaded_y_pred = load_y_pred(filename)

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(x_test, data['datasetTargets'][2],
                                            batch_size=128, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
                                                        accuracy * 100))

def simulateNN(model, x_test):
    y_pred = model.predict(x_test, batch_size=128, verbose=1)
    y_pred = y_pred.argmax(1)
    return y_pred

def load_y_pred():
    y_pred = pickle.load(open(filename, "rb"))
    return y_pred

def normalize_input(x):
    pool = multiprocessing.Pool(8)
    for i in range(len(x)):
        x[i] = pool.map(scale,[x[i]])[0]
    return x

def scale(img):
    img = image_histogram_equalization(img)
    avg = np.average(img)
    if (avg) != 0:
        for i in range(len(img)):
            img[i] = (img[i] - avg) / (avg)
    else: print(len(img))
    return img

def image_histogram_equalization(image, n_bins=256):
    # get image histogram
    histogram, bins = np.histogram(image, n_bins, normed=True)
    cdf = histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    return np.interp(image.flatten(), bins[:-1], cdf)

if __name__ == "__main__":
    main()
