import pvtools
import numpy
import matplotlib.pyplot

# .pvp files are produced from  ../tests/BasicSystemTest/Release/BasicSystemTest -p input/LCA_CIFAR-One-Image.params -l LCA-CIFAR-run.log -t 2

#The readpvpfile() function loads the .pvp file into a dictionary with keys header, values, and time
input_data = pvtools.readpvpfile("output/Input.pvp")
recon_data = pvtools.readpvpfile("output/InputRecon.pvp")

# There are 401 frames in the data structure, corresponding to the initial time t=0 and the 400 timesteps of the run. This number agrees with the nbands field of the header. Each frame is 32-by-32 with three features, corresponding to the size of the CIFAR-10 images, which are 32-by-32 images with three color channels. These values appear in the header as the nx, ny, and nf fields.
input_header = input_data['header']
print("Input.pvp header:")
for k in input_header.keys():
    print(f'{k:10} = {input_header[k]}')

input_values = input_data['values']
print("Input.pvp values shape:")
print(input_values.shape)

input_time = input_data['time']
print("Input.pvp time:")
print(input_time)



input_frame = input_values[0]

recon_values = recon_data['values']
recon_frame = recon_values[0]


def save_weights(wgts_display, filename):
    wgts_display_8bit = numpy.uint8(wgts_display * 127.5 + 127.500001)
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(wgts_display_8bit)
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close()
    print(f"Saved {filename}")

def save_frame(input_values, recon_values, frame_idx, filename):
    input_frame = input_values[frame_idx]
    recon_frame = recon_values[frame_idx]
    concatenated = numpy.vstack([input_frame, recon_frame])
    concat_normalized = (concatenated - numpy.min(concatenated)) / (numpy.max(concatenated) - numpy.min(concatenated))
    concat_8bit = numpy.uint8(concat_normalized * 255)
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(concat_8bit)
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close()
    print(f"Saved {filename}")
#Let's see what the image and reconstruction look like at the initial time.
save_frame(input_values, recon_values, 0, "frame_0.png")
save_frame(input_values, recon_values, 400, "frame_400.png")

#The dictionary of features is in the connection "LeakyIntegratorToInputError" and the evolution of the dictionary is written to the output/LeakyIntegratorToInputError.pvp file. Let's see how the weights are store

weights_data = pvtools.readpvpfile("output/LeakyIntegratorToInputError.pvp")
weights_header = weights_data['header']

for k in weights_header.keys():
    print(f'{k:10} = {weights_header[k]}')

weights_data['time']
print("Weights.pvp time:")
print(weights_data['time'])

weights_values = weights_data['values']
print("Weights.pvp values shape:")
print(weights_values.shape)

weights = weights_data['values'][0, 0]
print("Weights.pvp weights shape:")
print(weights.shape)
print("Weights.pvp weights min:")
print(numpy.min(weights))
print("Weights.pvp weights max:")
print(numpy.max(weights))

print("Weights.pvp arranged shape:")
wgts_display = pvtools.arrangedictionary(weights)
print(wgts_display.shape)

print("Weights.pvp arranged min:")
print(numpy.min(wgts_display))
print("Weights.pvp arranged max:")
print(numpy.max(wgts_display))

save_weights(wgts_display, "weights.png")

# The objective function is calculated by a probe, named TotalEnergyProbe in the params file. The probe produces the TotalEnergyProbe_batchElement_0.txt text file in the output directory.
probedata = numpy.loadtxt('output/TotalEnergyProbe_batchElement_0.txt', delimiter=',', skiprows=1)
probe_data_shape = probedata.shape
print(probe_data_shape)

timestamps = probedata[:, 0]
energy = probedata[:, 2]

def save_plot(x, y, title, filename):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(x, y)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close()
    print(f"Saved {filename}")

save_plot(timestamps, energy, 'Total Energy', "total_energy.png")


sparsity = pvtools.readlayerprobe(
    probe_name='SparsityProbe',
    directory='output',
    batch_element=0)
#sparsity['values'].shape (401, 1)
inputErrorL2Norm = pvtools.readlayerprobe(
    probe_name='InputErrorL2NormProbe',
    directory='output',
    batch_element=0)
#inputErrorL2Norm['values'].shape (401, 1)

save_plot(sparsity['time'], sparsity['values'].flatten(), 'Leaky Integrator L1-norm', "sparsity.png")
save_plot(inputErrorL2Norm['time'], inputErrorL2Norm['values'].flatten(), 'Input Reconstruction Error L2-norm', "input_error_l2norm.png")

# To create the gif with the animated_layer.py provided by OpenPV:
# import subprocess
# import os
# subprocess.run([
#     'python', os.path.join(os.environ['PV_SOURCEDIR'], 'tutorials/LCA-CIFAR/scripts/animate_layers.py'),
#     'output/Input.pvp', 'output/InputRecon.pvp', 'recon.gif'
# ], check=True)
import imageio

scale = 8  # 32x32 -> 256x256
gif_frames = []
breakpoint()  # Set a breakpoint here to inspect input_values and recon_value
for i in range(input_values.shape[0]):
    frame = numpy.vstack([input_values[i], recon_values[i]])
    minval, maxval = numpy.min(frame), numpy.max(frame)
    frame = frame if numpy.isclose(minval, maxval) else (frame - minval) / (maxval - minval)
    frame_8bit = numpy.uint8(frame * 255)
    gif_frames.append(numpy.kron(frame_8bit, numpy.ones((scale, scale, 1), dtype=numpy.uint8)))

imageio.mimsave('recon.gif', gif_frames)
print("Saved recon.gif")
