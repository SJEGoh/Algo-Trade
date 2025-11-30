# Algo-Trade


I am woefully underqualified to make an algorithmic trader, but I'm still going to give it my best shot over this holiday. 

The strategy I will be aiming to build is a pairs trading strategy (the choices of specific stocks will be handled by a different program). 

Pairs trading depends on multiple assumptions:
- Should be stationary

The concept is pretty simple. Take two assets that are similar in 

Unfortunately, real world stocks are generally not stationary. This can be mitigated however, first by finding two stocks that are cointegrated, and using a bayesian change-point detection model to change parameters in real time to shifting regimes. 

This model will be a proof of concept until I do actually get around to making the one that can choose cointegrated stocks. For now we will use __ and __. 

Angle granger method?

https://www.reddit.com/r/algotrading/comments/v7pchq/bayesian_hierarchical_models_for_algorithmic

