# PRC-data-challenge

Code for anyone interested
I dont care for rankings so comments are sparse. Good luck!

Thanks to the people who organized the challenge.


Some afterthoughts and ideas.

More knowledge about aviation in general required
*Many participants (not including me) are from the aviation scene, but who/what determines TOW? Pilots? Planners? Passengers? I dont know anything of these groups of people or systems.


Use Traffic library for some of the data wrangling instead of doing it by hand

Runway database
*Runway length & altitude
**Use traffic package to determine which runway is used (functions takeoff_from_runway & on_runway)
**https://traffic-viz.github.io/quickstart.html#making-maps
*What about nearby obstacles and noise limits?

Check different types of models
*I used linear regression and XGBoost

Climb estimation
*Trade-off between time and/or fuel usage.
**e.g. business class / first class might want faster while economy might prefer less fuel usage
**Where did the Concorde go?

Other stuff:
*as always, better data (complete & correct datasets mostly)

*Use engine count & other stuff in FAA database, more = better
*Less (big) categoricals , less = more
**What is the 'why' behind ades/adep/etc parameters?

*Median, Max & means for parameters for tuning, simple to add

*amount of airplanes per type in existence

*previous flights & next flights taken into account (preload fuel because next flight is uneconomical or unavailable?)
**Did do some quick testing here, I bet people can do better
**match ades/adep/timestamps
**Deviations from regular flights

*Time between flights (pilots sleep / same pilots / time to refuel)
*Fix altitude errors better 
**add start & end airports altitude instead of only time

*airlines flying certain paths a lot might have cheaper/better facilities or knowledge than other airlines

*group planes with little data with similar MTOW
