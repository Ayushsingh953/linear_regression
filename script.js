//CREATES A INPUT ARRAY THAT CONTAINS NUMBER FROM 1 TO 20
const INPUTS = [
  0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12,
  0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25,
  0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38,
  0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51,
  0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64,
  0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
  0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
  0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1,
];
// for (let n = 1; n <= 20; n++) {
//   INPUTS.push(n);
// }

//GENERATE OUTPUTS THAT ARE SIMPLY EACH INPUTS MULTIPLIED BY ITSELF.
//TO GENERATE SOME NON_LINEAR DATA.
const OUTPUTS = [
  5.37e-14, 7.49e-14, 1.05e-13, 1.46e-13, 2.04e-13, 2.85e-13, 3.98e-13,
  5.56e-13, 7.77e-13, 1.09e-12, 1.52e-12, 2.12e-12, 2.96e-12, 4.13e-12,
  5.76e-12, 8.04e-12, 1.12e-11, 1.56e-11, 2.18e-11, 3.04e-11, 4.23e-11,
  5.89e-11, 8.2e-11, 1.14e-10, 1.59e-10, 2.21e-10, 3.07e-10, 4.26e-10, 5.91e-10,
  8.19e-10, 1.13e-9, 1.57e-9, 2.17e-9, 3.0e-9, 4.15e-9, 5.72e-9, 7.89e-9,
  1.09e-8, 1.49e-8, 2.05e-8, 2.81e-8, 3.85e-8, 5.26e-8, 7.17e-8, 9.77e-8,
  1.33e-7, 1.8e-7, 2.43e-7, 3.28e-7, 4.4e-7, 5.88e-7, 7.82e-7, 1.04e-6, 1.36e-6,
  1.78e-6, 2.31e-6, 2.96e-6, 3.76e-6, 4.73e-6, 5.88e-6, 7.21e-6, 8.73e-6,
  1.04e-5, 1.23e-5, 1.43e-5, 1.65e-5, 1.87e-5, 2.1e-5, 2.34e-5, 2.57e-5,
  2.81e-5, 3.05e-5, 3.28e-5, 3.5e-5, 3.73e-5, 3.94e-5, 4.15e-5, 4.35e-5,
  4.55e-5, 4.74e-5, 4.92e-5, 5.1e-5, 5.27e-5, 5.43e-5, 5.59e-5, 5.74e-5,
  5.88e-5, 6.02e-5, 6.16e-5, 6.29e-5, 6.41e-5, 6.53e-5, 6.64e-5, 6.75e-5,
  6.86e-5, 6.96e-5, 7.06e-5, 7.15e-5, 7.24e-5, 7.33e-5, 7.4e-5,
];
// for (let n = 0; n < INPUTS.length; n++) {
//   OUTPUTS.push(INPUTS[n] * INPUTS[n]);
// }

// Input feature Array of Arrays needs 1D tensor to store.

const INPUTS_TENSOR = tf.tensor1d(INPUTS);

// Output can stay 1 dimensional.

const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

// Function to take a Tensor and normalize values

// with respect to each column of values contained in that Tensor.

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    // Find the minimum value contained in the Tensor.

    const MIN_VALUES = min || tf.min(tensor, 0);

    // Find the maximum value contained in the Tensor.

    const MAX_VALUES = max || tf.max(tensor, 0);

    // Now subtract the MIN_VALUE from every value in the Tensor

    // And store the results in a new Tensor.

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // Calculate the range size of possible values.

    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    // Calculate the adjusted values divided by the range size as a new Tensor.

    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });

  return result;
}

// Normalize all input feature arrays and then

// dispose of the original non normalized Tensors.

const FEATURE_RESULTS = normalize(INPUTS_TENSOR);

console.log("Normalized Values:");

FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log("Min Values:");

FEATURE_RESULTS.MIN_VALUES.print();

console.log("Max Values:");

FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();

//CREATE AND DEFINE MODEL ARCHITECTURE
const model = tf.sequential();

//WE WILL USE ONE DENSE LAYER WITH 3 NEURON(UNITS) AND AN INPUT
//OF ONE INPUT FEATURE
model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: "relu" }));

//ADD ANOTHER DENSE LAYER OF 1 NEURON THAT WILL BE CONNECTED TO THE FIRST INPUT LAYER ABOVE.
model.add(tf.layers.dense({ units: 1 }));

model.summary();
train();

async function train() {
  const LEARNING_RATE = 0.001; // Choose a learning rate that’s suitable for the data we are using.

  // Compile the model with the defined learning rate and specify a loss function to use.

  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),

    loss: "meanSquaredLogarithmicError",
  });

  // Finally do the training itself.

  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUTS_TENSOR,
    {
      callbacks: { onEpochEnd: logProgress },

      shuffle: true, // Ensure data is shuffled in case it was in an order

      batchSize: 10, // As we have a lot of training data, batch size is set to 64.

      epochs: 1000, // Go over the data 10 times!
    }
  );

  OUTPUTS_TENSOR.dispose();

  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  console.log(
    "Average error loss: " +
      Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );

  evaluate(); // Once trained evaluate the model.
}

function evaluate() {
  // Predict answer for a single piece of data.

  tf.tidy(function () {
    let newInput = normalize(
      tf.tensor1d([1]),
      FEATURE_RESULTS.MIN_VALUES,
      FEATURE_RESULTS.MAX_VALUES
    );

    let output = model.predict(newInput.NORMALIZED_VALUES);

    output.print();
  });

  // Finally when you no longer need to make any more predictions,

  // clean up the remaining Tensors.

  FEATURE_RESULTS.MIN_VALUES.dispose();

  FEATURE_RESULTS.MAX_VALUES.dispose();

  model.dispose();

  console.log(tf.memory().numTensors);
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, Math.sqrt(logs.loss));
}
