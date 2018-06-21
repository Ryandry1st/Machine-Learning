# pretty much an optimization problem. Uses crazy lagrangian multipliers
# look for where the mag of w is a minimum and the distance between the two vectors
# which is b, is a maximum
# This is a convex problem for optimization
# needs an idea of a starting value and a continued checking of increasing or decreasing

# useful optimization library is cvxopt or convex optimization
# another one is libsvm for svm optimization, used in sci-kit learn

# good book by stanford about convex optimization
# another option is SMO by microsoft, good for large data optimization


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self, visualization=True): # allows us to graph it high performance
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'} # red is plus, black is neg
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1) # makes one subplot 1 by 1

    #train based on data
    def fit(self, data):
        self.data = data
        opt_dict = {}
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1],]

        #good idea to get the max values rather than create a new list
        all_data = []

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
                    
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01,
                      # becomes expensive here
                      self.max_feature_value * 0.001,
                      ]

        # very very expensive! Not as valuable to get precise vs w
        b_range_multiple = 5
        
        b_multiple = 5
        optimum_factor = 10
        # first element in w, this saves processing by a lot
        latest_optimum = self.max_feature_value*optimum_factor
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            # stays false until there are no other steps to take, can do because this is a convex problem
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple, step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weak link in SVM fundamentals
                        # SMO tries to fix this, but it is still a huge data package
                        for yi in self.data: 
                            for xi in self.data[yi]: #yi(xi.w+b)>=1
                                if not yi*(np.dot(w_t, xi)+b )>= 1:
                                    found_option = False # could break and leave this one
                                    

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                            
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                    
                else:
                    w = w- step #does unit vector multiplication with step to the unit vector

            magnitudes = sorted([n for n in opt_dict])
            opt_choice = opt_dict[magnitudes[0]] #smallest magnitude of w

            # opt_dict looks like {||W|| : [w, b]}
            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step*2
            #optimum value is when yi(xi.w+b) = 1 for support vector + and -1 for minus group

        for yi in self.data:
            for xi in self.data[yi]:
                print(xi, ':', yi*(np.dot(self.w,xi)+self.b))
                
                                           

    def predict(self, features):
        # sign of (x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
            
        return classification


    #purely for humans to see, not useful for the calculation
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane is v = x.w+b
        # gives hyperplane values where v is the one we want
        # +sv = 1
        # -sv = -1
        # dec = 0
        def hyperplane(x,w,b, v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1) #just makes it easier to see edge points
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (x.w+b) = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()
        


data_dict = {-1:np.array([[1,7],[2,8], [3,8],]), 1:np.array([[5,1],[6,-1],[7,3],])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,-8]]

for p in predict_us:
    svm.predict(p)
svm.visualize()




