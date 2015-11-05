# Pedestrian dection and Tracking

## Implement pedestrian detection and tracking by Hog feature and Kalman filter

### Detect by Hog feature
[Histograms of Oriented Gradients(HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) feature extraction.
* Obtaining the human gradient information feature according to HoG operator.
* With overlapping block moving, the feature is a histograms of 4000 dimensions.

![Hog1](/features/capture1.png) ![Hog2](/features/capture2.png)
![Hog3](/features/capture3.png) ![Hog4](/features/capture4.png)

* The extracted features are use by [SVM]() to detection pedestrian candidates.

### Tracking by Kalman filter
* Using [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) to predict the movement of pedestrian.
* The movement vectors look like as following figures.

![movement1](/figures/capture8.png)
![movement2](/figures/capture9.png)
![movement3](/figures/capture10.png)