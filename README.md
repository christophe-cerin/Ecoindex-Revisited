# Ecoindex-Revisited

In this project, we revisit the calculation method of the EcoIndex metric. This metric has been proposed to evaluate its absolute environmental performance from a given URL using a score out of 100 (higher is better). Our motivation comes from the fact that the calculation is based on both prior quantile calculations and weightings. We propose keeping only the weighting mechanism corresponding to documented and regularly available figures of the proportional breakdown of ICT's carbon footprint. 

This way, we will be able to follow, from year to year, the evolution of web requests from a carbon footprint point of view. For a URL, our new calculation method takes as parameters three weights and the three typical values of the EcoIndex (DOM size, number of HTTP/HTTPS requests, KB transferred) and returns an environmental performance score. 

We develop several ways to compute the score based on our new hypothesis, either using learning techniques (Locality Sensitive Hashing, K Nearest Neighbor) or matrix computation constitutes the project's first contribution. The second contribution corresponds to an experimental study that allows us to estimate the differences in results between the methods. The whole work allows us to observe the environmental performance of the WEB in a more generic way than with the initial method. 

Indeed, the initial process requires recalculating each quantile according to the value of the chosen weights. It is, therefore, necessary to launch a benchmark, the HTTP archive, for example, at each new weighting. Our approaches do not require a systematic switch to a benchmark; thus, it is more generic than the previously known one.
