# How learning works: prediction, error, and improvement

**Prerequisites:** [AI mental models](ai-mental-models.md)  
**Outcome:** You can trace a prediction through a loss, an update, and an evaluation split.

## The central loop

Learning is not the model “thinking harder.” It is a repeated engineering loop:

```text
predict -> compare with the target -> measure error -> adjust parameters -> repeat
```

For a simple line, the model might predict delivery time:

\[
\hat{y} = wx + b
\]

Here, \(x\) is an input such as message length, \(\hat{y}\) is the prediction, and \(w\) and \(b\) are parameters.

If the true delivery time is \(y\), one simple error measure is squared error:

\[
loss = (\hat{y} - y)^2
\]

The loss is not a moral judgment. It is a number that tells the training procedure which parameter settings are more useful for its chosen objective.

## A numerical walk-through

Suppose `x = 2`, the true answer is `y = 10`, and we start with `w = 3`, `b = 1`.

```text
prediction = 3 × 2 + 1 = 7
error      = 7 - 10 = -3
loss       = (-3)² = 9
```

The model is under-predicting. An optimizer uses the direction and size of the error to propose a small parameter update. After many examples, a good set of parameters should make predictions useful on new examples too.

## The important separation

The training set is used to adjust parameters. The validation set helps choose among approaches. The test set is held back for the final estimate. If test examples influence repeated decisions, the test set is no longer an honest final check.

!!! warning "Memorization is not learning"
    A model can have low training loss by remembering examples and still fail on new inputs. Generalization—working beyond the exact examples—is the behavior we care about.

## Android bridge

The loop resembles performance tuning, but the object being changed is different. In Android, you might profile a slow screen, change an implementation, rerun the benchmark, and compare a metric. In ML, the training procedure changes parameters, and evaluation tells you whether the new parameters generalize.

The engineering discipline is similar: define the measurement, control the experiment, and investigate regressions instead of celebrating a single number.

## Exercises

1. With `w = 2`, `b = 0`, `x = 4`, and `y = 10`, calculate the prediction and squared loss.
2. Give one reason a model can improve training loss while becoming worse for real users.
3. Explain why the test set should not be used to choose the best of twenty model versions.

<details>
<summary>Solutions</summary>

1. Prediction is `8`; error is `-2`; squared loss is `4`.
2. It may overfit, or the training data may not represent production users.
3. Repeatedly choosing based on the test set makes it part of the decision process, so the final score becomes optimistic.

</details>

**Next topic:** [Python and reproducibility](python-reproducibility.md)

