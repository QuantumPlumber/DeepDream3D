import numpy as np 
import math
import argparse

def get_parser():
	parser = argparse.ArgumentParser(conflict_handler='resolve')

	# --------------- specify yml config -------------------------------

	# parser.add_argument("--yaml_config", action="store", dest="yaml_config", default=None, type=str,
	#                    help="Optionally specify parameters with a yaml file. YAML file overrides command line args")

	# --------------- specify model architecture -------------------------------

	parser.add_argument("--ae", action='store_true', dest="ae", default=False, help="True for ae [False]")
	parser.add_argument("--svr", action='store_true', dest="svr", default=False, help="True for svr [False]")

	# --------------- specify model mode -------------------------------

	parser.add_argument("--train", action='store_true', dest="train", default=False,
						help="True for training, False for testing")
	parser.add_argument("--getz", action='store_true', dest="getz", default=False,
						help="True for getting latent codes")
	parser.add_argument("--interpol", action='store_true', dest="interpol", default=False,
						help="True for interpolation")
	parser.add_argument("--deepdream", action='store_true', dest="deepdream", default=False,
						help="True for deepdream")

	# --------------- training -------------------------------

	parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int,
						help="Voxel resolution for coarse-to-fine training [64]")
	parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
	parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int,
						help="Iteration to train. Either epoch or iteration need to be zero [0]")
	parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float,
						help="Learning rate for adam [0.00005]")
	parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float,
						help="Momentum term of adam [0.5]")

	# --------------- testing -------------------------------

	parser.add_argument("--start", action="store", dest="start", default=0, type=int,
						help="In testing, output shapes [start:end]")
	parser.add_argument("--end", action="store", dest="end", default=16, type=int,
						help="In testing, output shapes [start:end]")

	# --------------- Data and Directories -------------------------------

	parser.add_argument("--R2N2_dir", action="store", dest="R2N2_dir", default="/shapenet",
						help="R2N2_dir directory")
	parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img",
						help="The name of dataset")
	parser.add_argument("--splitfile_dir", action="store", dest="splitfile",
						default="data/metadata/all_vox256_img_test.txt",
						help="The name of dataset")
	parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint",
						help="Directory name to save the checkpoints [checkpoint]")
	parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/",
						help="Root directory of dataset [data]")
	parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/",
						help="Directory name to save the image samples [samples]")
	parser.add_argument("--interpol_directory", action="store", dest="interpol_directory", default=None,
						help="First Interpolation latent vector")

	# --------------- Interpolation -------------------------------

	parser.add_argument("--interpol_z1", action="store", dest="interpol_z1", type=int, default=0,
						help="First Interpolation latent vector")
	parser.add_argument("--z1_im_view", action="store", dest="z1_im_view", type=int, default=23,
						help="First image number to deep dream with")
	parser.add_argument("--interpol_z2", action="store", dest="interpol_z2", type=int, default=1,
						help="Second Interpolation latent vector")
	parser.add_argument("--z2_im_view", action="store", dest="z2_im_view", type=int, default=23,
						help="Second image number to deep dream with")
	parser.add_argument("--interpol_steps", action="store", dest="interpol_steps", type=int, default=5,
						help="number of steps to take between values")

	# --------------- deepdream -------------------------------

	# dreaming uses the interpolation targets from interpolation as well as the number of steps.

	parser.add_argument("--layer_num", action="store", dest="layer_num", default=3, type=int,
						help="activation layer to maximize")
	parser.add_argument("--dream_rate", action="store", dest="dream_rate", default=.01, type=float,
						help="dream update rate")
	parser.add_argument("--beta", action="store", dest="beta", default=1e-9, type=float,
						help="scaling for style (gram matrix) loss")
	parser.add_argument("--annealing_rate", action="store", dest="annealing_rate", default=1, type=int,
						help="annealing rate")


	return parser

def write_ply_point(name, vertices):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	fout.close()


def write_ply_point_normal(name, vertices, normals=None):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("property float nx\n")
	fout.write("property float ny\n")
	fout.write("property float nz\n")
	fout.write("end_header\n")
	if normals is None:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
	else:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
	fout.close()


def write_ply_triangle(name, vertices, triangles):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()


def sample_points_triangle(vertices, triangles, num_of_points):
	epsilon = 1e-6
	triangle_area_list = np.zeros([len(triangles)],np.float32)
	triangle_normal_list = np.zeros([len(triangles),3],np.float32)
	for i in range(len(triangles)):
		#area = |u x v|/2 = |u||v|sin(uv)/2
		a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
		x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
		ti = b*z-c*y
		tj = c*x-a*z
		tk = a*y-b*x
		area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
		if area2<epsilon:
			triangle_area_list[i] = 0
			triangle_normal_list[i,0] = 0
			triangle_normal_list[i,1] = 0
			triangle_normal_list[i,2] = 0
		else:
			triangle_area_list[i] = area2
			triangle_normal_list[i,0] = ti/area2
			triangle_normal_list[i,1] = tj/area2
			triangle_normal_list[i,2] = tk/area2
	
	triangle_area_sum = np.sum(triangle_area_list)
	sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

	triangle_index_list = np.arange(len(triangles))

	point_normal_list = np.zeros([num_of_points,6],np.float32)
	count = 0
	watchdog = 0

	while(count<num_of_points):
		np.random.shuffle(triangle_index_list)
		watchdog += 1
		if watchdog>100:
			print("infinite loop here!")
			return point_normal_list
		for i in range(len(triangle_index_list)):
			if count>=num_of_points: break
			dxb = triangle_index_list[i]
			prob = sample_prob_list[dxb]
			prob_i = int(prob)
			prob_f = prob-prob_i
			if np.random.random()<prob_f:
				prob_i += 1
			normal_direction = triangle_normal_list[dxb]
			u = vertices[triangles[dxb,1]]-vertices[triangles[dxb,0]]
			v = vertices[triangles[dxb,2]]-vertices[triangles[dxb,0]]
			base = vertices[triangles[dxb,0]]
			for j in range(prob_i):
				#sample a point here:
				u_x = np.random.random()
				v_y = np.random.random()
				if u_x+v_y>=1:
					u_x = 1-u_x
					v_y = 1-v_y
				ppp = u*u_x+v*v_y+base
				
				point_normal_list[count,:3] = ppp
				point_normal_list[count,3:] = normal_direction
				count += 1
				if count>=num_of_points: break

	return point_normal_list