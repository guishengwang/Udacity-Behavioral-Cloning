{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D\n",
    "\n",
    "\n",
    "# Define model\n",
    "model = Sequential()\n",
    "model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(75, 320, 3)))\n",
    "model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=\"elu\"))\n",
    "model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=\"elu\"))\n",
    "model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=\"elu\"))\n",
    "model.add(Conv2D(64, (3, 3), activation=\"elu\"))\n",
    "model.add(Conv2D(64, (3, 3), activation=\"elu\"))\n",
    "model.add(Dropout(0.8))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(100))\n",
    "model.add(Dense(50))\n",
    "model.add(Dense(10))\n",
    "model.add(Dense(1))\n",
    "model.compile(optimizer='adam', loss='mse')\n",
    "\n",
    "# Train model\n",
    "X_train = np.load('X_train.npy')\n",
    "y_train = np.load('y_train.npy')\n",
    "model.fit(X_train, y_train, epochs=5, validation_split=0.1)\n",
    "\n",
    "# Save model\n",
    "model.save('model.h5')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
