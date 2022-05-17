import matplotlib.pyplot as plt

# plt.imshow(sim_obs1[0])
# plt.colorbar()

# plt.imshow(sim_obs2[0])

# plt.imshow(sim_obs_new[0][0])



#define two arrays for plotting
A = [3, 5, 5, 6, 7, 8]
B = [12, 14, 17, 20, 22, 27]

#create scatterplot, specifying marker size to be 40
plt.scatter(A, B, s=40)

#add arrow to plot
plt.arrow(x=4, y=18, dx=2, dy=5, width=.08) 
  
#display plot 
plt.show()