# airbnb_occupancy
Prediction of occupancy of accommodation listed under airbnb for the company Transparent Intelligence

The dataset consists of the pricing points (availability and nightly price) for a set of airbnb listings in a particular zipcode. The idea is that we can calculate the occupancy, that represents the amount of booked rooms for that zipcode at a particular pricing date (checkin date), at a particular time (scraping date). We would like to predict the final occupancy (occupancy where pricing date is the same as scraping date). We assume that occupancy = 1 - availability, that is the mean(available) where true is 1 and false is 0.

I took the data and made two very simple forecasts. One using a linear regression (basically just a straight line through the points) and one which recognise the seasonal and trendy nature of the data - a type of the ARIMA models. 
A general note is that I am using the same dataset for training and testing. With more data and in a production context, validation techniques should be used to reduce the over-fit. But here I am simply showing the use of two standard methods of time-series forecast. 

I made several findings while setting up these models:

ARIMA model whose parameters were regularly updated could actually be good at predicting the next day and month values. The series is strongly seasonal as one would expect with leisure activities and has a slight downward trend  as shown below so that a SARIMA model ( a seasonal ARIMA) makes sense. If the trend is actually a coincidence and disappears in larger sample or if more noise starts to appear around the trend, then the SARIMA can very well accommodate for that so that it is actually pretty robust (resists well to time). It would also be a transparent model with a trend and a seasonal part, which one can intuitively recognise as making sense or not, which is a good thing to detect problems when updating it. 

There is an interesting question around the use of scraping date data versus check-in date. In the simple version I am attaching here I have only used check-in dates data to predict occupancy at the time of check-in. It is clear however that occupancy before check-in is also a predictor of occupancy at check-in date so that it would be good to incorporate them as predictors as well. It is very possible and an interesting task. But it is actually a whole project on its own so I have sort of started to think about it but have kept it more simple for the example here. 

I also have similar comments about price. It is potentially an interesting predictor of occupancy but there are potential issues with causation: is occupancy driving price or price driving occupancy? There is a class of model called "co-integration" that deals well with this sort of problem, but again something to look at separately. 
The series of occupancy at check-in dates: 

see image in notebook

A first linear, straight line, prediction, which shows the slightly downward trend: 

see image in notebook

A very simple SARIMA, with no optimisation of the parameters or stationarity treatment at all. It actually captures decently the overall shape except for the first point. 

see image in notebook

I am attaching the code. It is in a python notebook but I have aimed to write it so that it can be run from the terminal / as a script as well.

A last point concerning more advanced techniques, such as long short-term Memory (LSTM). I feel like these methods could be very useful but:
It is a good idea to use simple things that work before doing more advanced ones. I'd actually recommend using an ensemble of models, for example, one SARIMA, one GARCH potentially and deep learning ones. The end prediction could be the median of each. 
