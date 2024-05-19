import os
import numpy as np
import matplotlib.pyplot as plt
# Sensor size
img_size = (346, 240)

# Call the function to read data    
filename_sub = "EV_Eye_dataset\\raw_data\\Data_davis\\user1\\left\\session_1_0_1\\events\\events.txt"
# Brightness increment image (Balance of event polarities)
num_events = 10000
print("Brightness increment image: numevents =", num_events)


def extract_data(filename, max_samples=1000000):
    timestamp = []
    x = []
    y = []
    pol = []
    samples_read = 0
    with open(filename, 'r') as infile:
        for line in infile:
            words = line.split()
            if len(words) >= 4:
                timestamp.append(float(words[0]))
                x.append(int(words[1]))
                y.append(int(words[2]))
                pol.append(int(words[3]))
                samples_read += 1
                if samples_read >= max_samples:
                    break
    return timestamp, x, y, pol
timestamp, x, y, pol = extract_data(filename_sub)

# num_events = len(timestamp)
# img = np.zeros(img_size, np.int)
# for i in range(len(timestamp)):
#     # Check if coordinates are within bounds
#     if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
#         # Need to convert the polarity bit from {0,1} to {-1,+1} and accumulate
#         img[y[i], x[i]] += (2 * pol[i] - 1)


# fig = plt.figure()
# fig.suptitle('Balance of event polarities')
# maxabsval = np.amax(np.abs(img))
# plt.imshow(img, cmap='gray', clim=(-maxabsval, maxabsval))
# plt.xlabel("x [pixels]")
# plt.ylabel("y [pixels]")
# plt.colorbar()
# 

# 2D Histograms of events, split by polarity
img_pos = np.zeros(img_size, np.int)
img_neg = np.zeros(img_size, np.int)
for i in range(len(timestamp)):  # Adjust the loop range to the number of events
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:  # Check if coordinates are within bounds
        if pol[i] > 0:
            img_pos[y[i], x[i]] += 1
        else:
            img_neg[y[i], x[i]] += 1


fig = plt.figure()
fig.suptitle('Histogram of positive events')
plt.imshow(img_pos)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


fig = plt.figure()
fig.suptitle('Histogram of negative events')
plt.imshow(img_neg)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


# Thresholded brightness increment image (Ternary image)
img = np.zeros(img_size, np.int)
for i in range(num_events):
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:  # Check if coordinates are within bounds
        img[y[i], x[i]] = (2 * pol[i] - 1)

fig = plt.figure()
fig.suptitle('Last event polarity per pixel')
plt.imshow(img, cmap='gray')
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sensor size
img_size = (240, 346)

# Call the function to read data    
filename_sub = "EV_Eye_dataset\\raw_data\\Data_davis\\user1\\left\\session_1_0_1\\events\\events.txt"
# Brightness increment image (Balance of event polarities)
num_events = 10000
print("Brightness increment image: numevents =", num_events)

# Read file with a subset of events
def extract_data(filename, max_samples=10000):
    timestamp = []
    x = []
    y = []
    pol = []
    samples_read = 0
    with open(filename, 'r') as infile:
        for line in infile:
            words = line.split()
            if len(words) >= 4:
                timestamp.append(float(words[0]))
                x.append(int(words[1]))
                y.append(int(words[2]))
                pol.append(int(words[3]))
                samples_read += 1
                if samples_read >= max_samples:
                    break
    return timestamp, x, y, pol

timestamp, x, y, pol = extract_data(filename_sub)

# What if we only use 3 values in the event accumulation image?
# Saturated signal: -1, 0, 1
# For example, store the polarity of the last event at each pixel
img = np.zeros(img_size, np.int)
for i in range(num_events):
    # Check if coordinates are within bounds
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
        img[y[i], x[i]] = (2 * pol[i] - 1)  # no accumulation; overwrite the stored value


# Display the ternary image
fig = plt.figure()
fig.suptitle('Last event polarity per pixel')
plt.imshow(img, cmap='gray')
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


# _____________________________________________________________________________
# Time surface (or time map, or SAE="Surface of Active Events")
num_events = len(timestamp)
print("Time surface: numevents = ", num_events)

img = np.zeros(img_size, np.float32)
t_ref = timestamp[-1] # time of the last event in the packet
tau = 0.03 # decay parameter (in seconds)
for i in range(num_events):
    # Check if coordinates are within bounds
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
        img[y[i], x[i]] = np.exp(-(t_ref - timestamp[i]) / tau)


fig = plt.figure()
fig.suptitle('Time surface (exp decay). Both polarities')
plt.imshow(img)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


# Time surface (or time map, or SAE), separated by polarity
sae_pos = np.zeros(img_size, np.float32)
sae_neg = np.zeros(img_size, np.float32)
for i in range(num_events):
    # Check if coordinates are within bounds
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
        if pol[i] > 0:
            sae_pos[y[i], x[i]] = np.exp(-(t_ref - timestamp[i]) / tau)
        else:
            sae_neg[y[i], x[i]] = np.exp(-(t_ref - timestamp[i]) / tau)


fig = plt.figure()
fig.suptitle('Time surface (exp decay) of positive events')
plt.imshow(sae_pos)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


fig = plt.figure()
fig.suptitle('Time surface (exp decay) of negative events')
plt.imshow(sae_neg)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


# Time surface (or time map, or SAE), using polarity as sign of the time map
sae = np.zeros(img_size, np.float32)
for i in range(num_events):
    # Check if coordinates are within bounds
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
        if pol[i] > 0:
            sae[y[i], x[i]] = np.exp(-(t_ref - timestamp[i]) / tau)
        else:
            sae[y[i], x[i]] = -np.exp(-(t_ref - timestamp[i]) / tau)


fig = plt.figure()
fig.suptitle('Time surface (exp decay), using polarity as sign')
plt.imshow(sae, cmap='seismic') # using color (Red/blue)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


# "Balance of time surfaces"
# Accumulate exponential decays using polarity as sign of the time map
sae = np.zeros(img_size, np.float32)
for i in range(num_events):
    # Check if coordinates are within bounds
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
        if pol[i] > 0:
            sae[y[i], x[i]] += np.exp(-(t_ref - timestamp[i]) / tau)
        else:
            sae[y[i], x[i]] -= np.exp(-(t_ref - timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay), balance of both polarities')
maxabsval = np.amax(np.abs(sae))
plt.imshow(sae, cmap='seismic', clim=(-maxabsval,maxabsval))
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


# Average timestamp per pixel
sae = np.zeros(img_size, np.float32)
count = np.zeros(img_size, np.int)
for i in range(num_events):
    # Check if coordinates are within bounds
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
        sae[y[i], x[i]] += timestamp[i]
        count[y[i], x[i]] += 1
    
# Compute per-pixel average if count at the pixel is >1
count [count < 1] = 1  # to avoid division by zero
sae = sae / count

fig = plt.figure()
fig.suptitle('Average timestamps regardless of polarity')
plt.imshow(sae)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


# Average timestamp per pixel. Separate by polarity
sae_pos = np.zeros(img_size, np.float32)
sae_neg = np.zeros(img_size, np.float32)
count_pos = np.zeros(img_size, np.int)
count_neg = np.zeros(img_size, np.int)
for i in range(num_events):
    # Check if coordinates are within bounds
    if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
        if pol[i] > 0:
            sae_pos[y[i], x[i]] += timestamp[i]
            count_pos[y[i], x[i]] += 1
        else:
            sae_neg[y[i], x[i]] += timestamp[i]
            count_neg[y[i], x[i]] += 1
# Compute per-pixel average if count at the pixel is >1
count_pos [count_pos < 1] = 1;  sae_pos = sae_pos / count_pos
count_neg [count_neg < 1] = 1;  sae_neg = sae_neg / count_neg

fig = plt.figure()
fig.suptitle('Average timestamps of positive events')
plt.imshow(sae_pos)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


fig = plt.figure()
fig.suptitle('Average timestamps of negative events')
plt.imshow(sae_neg)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()


# _____________________________________________________________________________
# 3D plot 
# Time axis in horizontal position

m = 2000 # Number of points to plot
print("Space-time plot and movie: numevents = ", m)

# Plot without polarity
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal') # only works for time in Z axis
ax.scatter(x[:m], timestamp[:m], y[:m], marker='.', c='b')
ax.set_xlabel('x [pix]')
ax.set_ylabel('time [s]')
ax.set_zlabel('y [pix] ')
ax.view_init(azim=-90, elev=-180) # Change viewpoint with the mouse, for example


# Plot each polarity with a different color (red / blue)
idx_pos = np.asarray(pol[:m]) > 0
idx_neg = np.logical_not(idx_pos)
xnp = np.asarray(x[:m])
ynp = np.asarray(y[:m])
tnp = np.asarray(timestamp[:m])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xnp[idx_pos], tnp[idx_pos], ynp[idx_pos], marker='.', c='b')
ax.scatter(xnp[idx_neg], tnp[idx_neg], ynp[idx_neg], marker='.', c='r')
ax.set(xlabel='x [pix]', ylabel='time [s]', zlabel='y [pix]')
ax.view_init(azim=-90, elev=-180)


# Transition between two viewpoints
num_interp_viewpoints = 60 # number of interpolated viewpoints
ele = np.linspace(-150,-180, num=num_interp_viewpoints)
azi = np.linspace( -50, -90, num=num_interp_viewpoints)

# Create directory to save images and then create a movie
dirName = 'tempDir'
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")
    
for ii in range(0,num_interp_viewpoints):
    ax.view_init(azim=azi[ii], elev=ele[ii])
    plt.savefig(dirName + "/movie%04d.png" % ii)

# Create a movie using ffmpeg static build (https://johnvansickle.com/ffmpeg/)
def createMovie():
    os.system("/home/ggb/Downloads/ffmpeg-4.2.2-i686-static/ffmpeg -r 20 -i " 
    + dirName  + "/movie%04d.png -c:v libx264 -crf 0 -y movie_new.mp4")

# Call the function to create the movie
createMovie()

# _____________________________________________________________________________
# Voxel grid

num_bins = 5
print("Number of time bins = ", num_bins)

t_max = np.amax(np.asarray(timestamp[:m]))
t_min = np.amin(np.asarray(timestamp[:m]))
t_range = t_max - t_min
dt_bin = t_range / num_bins # size of the time bins (bins)
t_edges = np.linspace(t_min,t_max,num_bins+1) # Boundaries of the bins

# Compute 3D histogram of events manually with a loop
# ("Zero-th order or nearest neighbor voting")
hist3d = np.zeros(img.shape+(num_bins,), np.int)
for ii in range(m):
    # Check if coordinates are within bounds
    if 0 <= y[ii] < img.shape[0] and 0 <= x[ii] < img.shape[1]:
        idx_t = int((timestamp[ii] - t_min) / dt_bin)
        if idx_t >= num_bins:
            idx_t = num_bins - 1  # only one element (the last one)
        hist3d[y[ii], x[ii], idx_t] += 1


# Compute 3D histogram of events using numpy function histogramdd
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogramdd.html#numpy.histogramdd

# Specify bin edges in each dimension
bin_edges = (np.linspace(0,img_size[0],img_size[0]+1), 
             np.linspace(0,img_size[1],img_size[1]+1), t_edges)
yxt = np.transpose(np.array([y[:m], x[:m], timestamp[:m]]))
hist3dd, edges = np.histogramdd(yxt, bins=bin_edges)

# Compute interpolated 3D histogram (voxel grid)
hist3d_interp = np.zeros(img.shape+(num_bins,), np.float64)
for ii in range(m-1):
    # Check if coordinates are within bounds
    if 0 <= y[ii] < hist3d_interp.shape[0] and 0 <= x[ii] < hist3d_interp.shape[1]:
        tn = (timestamp[ii] - t_min) / dt_bin  # normalized time, in [0,num_bins]
        ti = np.floor(tn - 0.5)  # index of the left bin
        dt = (tn - 0.5) - ti  # delta fraction
        # Voting on two adjacent bins
        if ti >= 0:
            hist3d_interp[y[ii], x[ii], int(ti)] += 1. - dt
        if ti < num_bins - 1:
            hist3d_interp[y[ii], x[ii], int(ti) + 1] += dt


# Compute interpolated 3D histogram (voxel grid) using polarity
hist3d_interp_pol = np.zeros(img.shape+(num_bins,), np.float64)
for ii in range(m-1):
    # Check if coordinates are within bounds
    if 0 <= y[ii] < hist3d_interp_pol.shape[0] and 0 <= x[ii] < hist3d_interp_pol.shape[1]:
        tn = (timestamp[ii] - t_min) / dt_bin  # normalized time, in [0,num_bins]
        ti = np.floor(tn - 0.5)  # index of the left bin
        dt = (tn - 0.5) - ti  # delta fraction
        # Voting on two adjacent bins
        if ti >= 0:
            hist3d_interp_pol[y[ii], x[ii], int(ti)] += (1. - dt) * (2 * pol[ii] - 1)
        if ti < num_bins - 1:
            hist3d_interp_pol[y[ii], x[ii], int(ti) + 1] += dt * (2 * pol[ii] - 1)


# Plot of the 3D histogram
fig = plt.figure()
fig.suptitle('3D histogram (voxel grid), zero-th order voting')
ax = fig.gca(projection='3d')
ax.voxels(hist3d) 
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
ax.view_init(azim=-90, elev=-180)


# Plot of the interpolated 3D histogram (voxel grid)
fig = plt.figure()
fig.suptitle('Interpolated 3D histogram (voxel grid)')
ax = fig.gca(projection='3d')
ax.voxels(hist3d_interp) 
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
ax.view_init(azim=-63, elev=-145)


# Plot of the interpolated 3D histogram (voxel grid) using polarity
fig = plt.figure()
fig.suptitle('Interpolated 3D histogram (voxel grid), including polarity')
ax = fig.gca(projection='3d')
ax.voxels(hist3d_interp_pol) 
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
ax.view_init(azim=-63, elev=-145)
plt.show()
