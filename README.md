# Building-Recommender-System-using-machine-learning-and-AI

Extrawork from user,
1. put a thumbsup/rate 1-10/ write a comment. not everyone likes to this, so we have sparse data which makes low quality recommendation system.
Also, some people are critical, cultural differences.
2. Implicit behavior: which links users are clicking, those are positive. click data is great, so much of it, no problem with sparsity. sometimes 
not reliable, people click by accident. things people purchase, is true indication of interests. Also things user consume, youtube can see how
many minitues user spent on something

*** hard part is to buil the databse for item similarities

**K-fold validation algo -> prevents overfitting
* we want to recommend people they havent seen and they might find interesting
** Accuracy metric:
mean absolute error (MAE)
Root mean square error(RMSE)
RMSE does not really matter, what matters which movies you put infront of your user and how they react. 
***evaluating top-n recommender
hit rate: if user actually hit like to a video. sum(all hits in top n from every users) / # of users. but measuting hit rate is tricky
We can't use the same train test or cross validation approach we used for measuring accuracy, because we're not measuring the accuracy on
individual ratings. We're measuring the accuracy of top-end lists for individual users. Now you could do the obvious thing and not split
things up at all, and just measure hit rate directly on top end recommendations, created by a recommender system, that was trained on all


of the data you have. But, technically, that's cheating. You generally don't want to evaluate a system using data that it was trained with.
I mean think about it. You could just recommend the actual top ten movies rated by each user, using the training data, and achieve a hit
rate of 100%. So a clever way around this is called leave-one-out cross validation. What we do is compute the top end recommendations for 
each user in our training data, and intentionally remove one of those items from that users training data. We then test our recommenders

system's ability to recommend that item that was left out in the top end results it creates for that user in the testing phase. So we 
measure our ability to recommend an item in a top end list for each user that was left out from the training data. That's why it's 
called leave-one-out. The trouble is, it's a lot harder to get one specific movie right, while testing, than to just get one of the
end recommendations. So hit rate with leave-one-out tends to be very small, and difficult to measure, unless you have a very large
data sets to work with. But it's a much more user-focused metric when you know your recommender system will be producing top end lists 
in the real world, which most of them do. A variation on hit rate is average reciprocal hit rate, or ARHR for short. This metric is just
like hit rate, but it accounts for where in the top end list your hits appear. So you end up getting more credit for successfully
recommending an item in the top slot, than in the bottom slot. Again, this is a more user-focused metric, since users tend to focus on 
the beginning of lists. The only difference is that instead of summing up the number of hits, we sum up the reciprocal rank of each hit.
So if we successfully predict a recommendation in slot three, that only counts as one-third. But a hit in slot one of our top end
recommendations receives the full weight of 1.0. Whether this metric makes sense for you, depends a lot on how your top end 
recommendations are displayed. If the user has to scroll or paginate to see the lower items in your top end list, then it makes sense to
penalize good recommendations that appear too low in the list, where the user has to work to find them. Another twist is cumulative hit
rank. Sounds fancy, but all it means is that we throw away hits if our predicted ratings below some threshold. The idea is that we 
shouldn't get credit for recommending items to a user that we think they won't actually enjoy. So in this example, if we had a cutoff of
three stars, we'd throw away the hits for the second and fourth items in these test results, and our hit rate metric wouldn't count 
them at all. Yet another way to look at hit rate, is to break it down by predicted rating score. It can be a good way to get an idea of
the distribution of how good your algorithm thinks recommended movies are, that actually get a hit. Ideally, you want to recommend movies
that they actually liked. And breaking down the distribution gives you some sense of how well you're doing in more detail. This is called
rating hit rate, or rHR, for short. So those are all different ways to measure the effectiveness of top end recommenders offline. The
world of recommender systems would probably be a little bit different if Netflix awarded the Netflix prize on hit rate, instead of RMSE.
It turns out that small improvements in RMSE can actually result in large improvements to hit rates, which is what really matters. But 
it also turns out that you can build recommender systems with great hit rates, but poor RMSE scores. And we'll see some of those later 
in this course. So RMSE and hit rate aren't always related.

Coverage, diversity, and novelty
- [Instructor] Accuracy isn't the only thing that matters with recommender systems. There are other things we can measure if they're important to us. For example, coverage. That's just the percentage of possible recommendations that your system is able to provide. Think about the movie lens data set of movie ratings we're using in this course. It contains ratings for several thousand movies, but there are plenty of movies in existence that it doesn't have ratings for. If you were using this data to generate recommendations on say, IMDB, then the coverage of this recommender system would be low, because IMDB has millions of movies in its catalog, not thousands. It's worth noting that coverage can be at odds with accuracy. If you enforce a higher quality threshold on the recommendations you make, then you might improve your accuracy at the expense of coverage. Finding the balance of where exactly you're better off recommending nothing at all can be delicate. Coverage can also be important to watch because it gives you a sense of how quickly new items in your catalog will start to appear in recommendations. When a new book comes out on Amazon, it won't appear in recommendations until at least a few people buy it, therefore establishing patterns with the purchase of other items. Until those patterns exist, that new book will reduce Amazon's coverage metric. Another metric is called diversity. You can think of this as a measure of how broad a variety of items your recommender system is putting in front of people. An example of low diversity would be a recommender system that just recommends the next books in a series that you've started reading, but doesn't recommend books from different authors or movies related to what you've read. This may seem like a subjective thing, but it is measurable. Many recommender systems start by computing some sort of similarity metric between items. So we can use these similarity scores to measure diversity. If we look at the similarity scores of every possible pair in a list of top end recommendations, we can average them to get a measure of how similar the recommended items in the list are to each other. We can call that measure S. Diversity is basically the opposite of average similarity, so we subtract it from one to get a number associated with diversity. It's important to realize that diversity, at least in the context of recommender systems, isn't always a good thing. You can achieve very high diversity by just recommending completely random items. But those aren't good recommendations by any stretch of the imagination. Unusually high diversity scores mean that you just have bad recommendations more often than not. You always need to look at diversity alongside metrics that measure the quality of the recommendations as well. Similarly, novelty sounds like a good thing, but often it isn't. Novelty is a measure of how popular the items are that you're recommending. And again, just recommending random stuff would yield very high novelty scores since the vast majority of items are not top sellers. Although novelty is measurable, what to do with it is in many ways subjective. There's a concept of user trust in a recommender system. People want to see at least a few familiar items in their recommendations that make them say yeah, that's a good recommendation for me. This system seems good. If you only recommend things people have never heard of, they may conclude that your system doesn't really know them, and they may engage less with your recommendations as a result. Also, popular items are usually popular for a reason. They're enjoyable by a large segment of the population, so you would expect them to be good recommendations for a large segment of the population who hasn't read or watched them yet. If you're not recommending some popular items, you should probably question whether your recommender system is really working as it should. This is an important point. You need to strike a balance between familiar popular items and what we call serendipitous discovery of new items the user has never heard of before. The familiar items establish trust with the user and the new ones allow the user to discover entirely new things that they might love. Novelty is important though because the whole point of recommender systems is to service items in what we call the long tail. Imagine this is a plot of the sales of items of every item in your catalog sorted by sales. So the number of sales or popularity is on the Y axis, and all the products are along the X axis. You almost always see an exponential distribution like this. Most sales come from a very small number of items, but taken together, the long tail makes up a large amount of sales as well. Items in that long tail, the yellow part in the graph, are items that cater to people with unique niche interests. Recommender systems can help people discover those items in the long tail that are relevant to their own unique niche interests. If you can do that successfully, then the recommendations your system makes can help new authors get discovered, can help people explore their own passions, and make money for whoever you're building the system for as well. Everybody wins. When done right, recommender systems with good novelty scores can actually make the world a better place. But again, you need to strike a balance between novelty and trust. As I said, building recommender systems is a bit of an art, and this is an example of why.
