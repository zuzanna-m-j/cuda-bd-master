#!/usr/bin/env python3

"""
Author:
Date:
File name:

Program description:

Generates input file to be run by gpu-tild code
Creates a polymer sequence using the HP0 proptein model

Features:

- generate according to a given sequence


***** extend by non-spherical monomers





"""


# imports

import random
import numpy as np
import copy
import os


# flags

EXPLICIT_SOLVENT = False
SALT = False
SEQUENCE = True
EXTENDED = False
COARSE = False
WLC = False
seq_file = 'sequence.seq'


file_name = f'/home/cappuccino/Downloads/cuda-bd-master/research/beta/AB-charged/input'

molecule_types = 1 # different protein chains

b_mon = 1.0 # size of the monomer
b_sol = b_mon*4 # solvent size
b_cat = b_sol/4
b_anion = b_sol
phi = [0.1] # volume fractions of chains
q = [1.0] # available charges
q_cat = 1.0
q_anion = -1.0

mon_mass = 1.0000
cat_mass = 1.0000/4
anion_mass = 1.000/4
drude_mass = 0.100
sol_mass = 1.000/2



# simulation steps

dim = 3
box_dim = [100.0000, 100.0000, 100.0000] # x y z
max_steps = 2000001
log_freq = 10000
binary_freq = 2000
traj_freq = 200000
b_len = b_mon/2
smear_len = b_mon
Nx = 125
Ny =125
Nz = 125

# COARSE

c_scale = 4.0

if COARSE == True:
    b_mon = c_scale * b_mon
    mon_mass = c_scale**3 * mon_mass

atom_count = 1
mol_count = 1
bond_count = 1
angle_count = 1

# Monomers

# id charge drude

H = [1, 1, 0]
B = [2, 1, 0]
P = [3, 0, 1]
A = [4, 0, 0]
D = [5, 1, 0]
W = [6, 0, 1]
Cation = [7, 1, 1]
Anion = [8, 1, 1]


types = [H,B,P,A,D, W, Cation, Anion]
masses = [mon_mass] * 4 + [drude_mass] + [sol_mass] + [cat_mass] + [anion_mass]

particle_types = len(types)
bond_types = 3
angle_types = 1


# chain composition

molecules = [[0,1,2]]

if SEQUENCE == True:
    seq = []
    with open(seq_file) as file:
        lines = file.readlines()
        for line in lines:
            for m in line:
                seq.append(m)
    N = [len(seq)]
else:
    N = [20]

# bonds - mon - mon
# mon- drude
# gaussian
# angle
# generate molecules

properties = [] # num mol_id type_id charge x y z drude
bonds = []
mol_angles = []
angles = []

# bond type-1 - mon-mon
# bond type-2 - mon-drude
# calculate number of molecules needed

# calculate "cube volume"

v_mon = b_mon**3
v_sol = b_sol**3

# calculate the needed number of molecules of each type
# len of list matches the molecule types

N_molecules = []

box_vol = box_dim[0] * box_dim[1] * box_dim[2]
for i in range(molecule_types):
    N_molecules.append(int(phi[i] * box_vol/(N[i]*v_mon)))
print(f"Volume fraction {N_molecules[0] * N[0] * v_mon/box_vol} {N_molecules[0]}")
N_cat = int(N_molecules[0] * 0.2)
N_anion = int(q_cat * N_cat//abs(q_anion))
v_cat = b_cat**3
v_anion = b_cat**3

phi_salt = (N_cat * v_cat + N_anion * v_anion)/box_vol
phi_sol = 1 - np.sum(phi) - phi_salt
N_sol = int((phi_sol * box_vol)//v_sol)

# generate random starting position for the molecule_types

for m_type in range(molecule_types):
    for m_num in range(N_molecules[m_type]):
        mol_ang = []
        for chain_pos in range(N[m_type]):
            if SEQUENCE == True:
                m = seq[chain_pos]
                # swith
                if m == 'H':
                    m = H[0] - 1
                elif m == 'B':
                    m = B[0] - 1
                elif m == 'P':
                    m = P[0] - 1
                elif m == 'A':
                    m = A[0] - 1
                else:
                    raise Exception("You have failed miserably, Zuzanna!")
            else:
                m = random.choice(molecules[m_type])
            props = [atom_count,mol_count,types[m][0]] #initialize an entry for a chosen type

            # id charge drude2
            #drude oscillator

            # if the drude flag is set to 1

            if types[m][2] == 1:

                #if the molecule is charged

                if types[m][1] == 1:
                    atom_charge =  np.random.choice([q[0],q[0]])

                    if atom_charge > 0:
                        drude_charge = -0.5 * atom_charge
                        atom_charge -=  drude_charge
                    else:
                        drude_charge = 0.5 * atom_charge
                        atom_charge -= drude_charge


                else:
                    atom_charge = q[0]/2
                    drude_charge = -atom_charge
                props.append(atom_charge)

                # start a the chain

                if chain_pos == 0:
                    for xyz in range(dim):
                        coord = np.random.uniform(0,box_dim[xyz])
                        props.append(coord)
                else:
                    if properties[-1][2] != D[0]:
                        theta = random.uniform(-np.pi, np.pi)
                        phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                        x = b_mon * np.cos(phi)*np.sin(theta) + properties[-1][4]
                        y = b_mon * np.sin(phi)*np.sin(theta) + properties[-1][5]
                        z = b_mon * np.cos(theta) + properties[-1][6]
                        props.append(x)
                        props.append(y)
                        props.append(z)

                        if EXTENDED == True and chain_pos > 2:
                            if properties[-2][2] != D[0]:
                                ind = -2
                            else:
                                ind = -3
                            d2 = properties[ind]
                            d0 = np.sqrt((x - d2[4])**2 + (y - d2[5])**2 + (z - d2[6])**2)
                            while (d0 < 0.5):
                                theta = random.uniform(-np.pi, np.pi)
                                phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                                x = b_mon * np.cos(phi)*np.sin(theta) + properties[-1][4]
                                y = b_mon * np.sin(phi)*np.sin(theta) + properties[-1][5]
                                z = b_mon * np.cos(theta) + properties[-1][6]
                                props.append(x)
                                props.append(y)
                                props.append(z)
                                d0 = np.sqrt((x - d2[4])**2 + (y - d2[5])**2 + (z - d2[6])**2)
                    else:
                        theta = random.uniform(-np.pi, np.pi)
                        phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                        x = b_mon * np.cos(phi)*np.sin(theta) + properties[-2][4]
                        y = b_mon * np.sin(phi)*np.sin(theta) + properties[-2][5]
                        z = b_mon * np.cos(theta) + properties[-2][6]
                        props.append(x)
                        props.append(y)
                        props.append(z)

                        if EXTENDED == True and chain_pos > 2:
                            if properties[-3][2] != D[0]:
                                ind = -3
                            else:
                                ind = -4
                            d2 = properties[ind]
                            d0 = np.sqrt((x - d2[4])**2 + (y - d2[5])**2 + (z - d2[6])**2)
                            while (d0 < 0.5):
                                theta = random.uniform(-np.pi, np.pi)
                                phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                                x = b_mon * np.cos(phi)*np.sin(theta) + properties[-1][4]
                                y = b_mon * np.sin(phi)*np.sin(theta) + properties[-1][5]
                                z = b_mon * np.cos(theta) + properties[-1][6]
                                props.append(x)
                                props.append(y)
                                props.append(z)
                                d0 = np.sqrt((x - d2[4])**2 + (y - d2[5])**2 + (z - d2[6])**2)
                
                # add atom properties to the list
                properties.append(copy.deepcopy(props))
                mol_ang.append(atom_count)

                # drude bond - type 2
                bonds.append([bond_count,2,atom_count,atom_count+1])
                bond_count += 1

                # regular bond - 1
                if chain_pos != (N[m_type]-1):
                    bonds.append([bond_count,1,atom_count,atom_count+2])
                    bond_count += 1

                # advance the atom count
                atom_count += 1

                # add the drude oscilator

                dx = np.random.choice([-1,1]) * b_mon/np.sqrt(3) + properties[-1][4]
                dy = np.random.choice([-1,1]) * b_mon/np.sqrt(3) + properties[-1][5]
                dz = np.random.choice([-1,1]) * b_mon/np.sqrt(3) + properties[-1][6]

                props = [atom_count,mol_count,D[0],drude_charge,dx,dy,dz]
                properties.append(copy.deepcopy(props))
                atom_count += 1

                # no drude oscillator
            else:
                if types[m][1] == 1:
                    atom_charge = np.random.choice([q[0],q[0]])
                else:
                    atom_charge = 0.0
                props.append(atom_charge)
                # start the chain
                if chain_pos == 0:
                    for xyz in range(dim):
                        coord = np.random.uniform(0,box_dim[xyz])
                        props.append(coord)
                else:
                    if properties[-1][2] != D[0]:
                        theta = random.uniform(-np.pi, np.pi)
                        phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                        x = b_mon * np.cos(phi)*np.sin(theta) + properties[-1][4]
                        y = b_mon * np.sin(phi)*np.sin(theta) + properties[-1][5]
                        z = b_mon * np.cos(theta) + properties[-1][6]
                        props.append(x)
                        props.append(y)
                        props.append(z)

                        if EXTENDED == True and chain_pos > 2:
                            if properties[-2][2] != D[0]:
                                ind = -2
                            else:
                                ind = -3
                            d2 = properties[ind]
                            d0 = np.sqrt((x - d2[4])**2 + (y - d2[5])**2 + (z - d2[6])**2)
                            while (d0 < 0.5):
                                theta = random.uniform(-np.pi, np.pi)
                                phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                                x = b_mon * np.cos(phi)*np.sin(theta) + properties[-1][4]
                                y = b_mon * np.sin(phi)*np.sin(theta) + properties[-1][5]
                                z = b_mon * np.cos(theta) + properties[-1][6]
                                props.append(x)
                                props.append(y)
                                props.append(z)
                                d0 = np.sqrt((x - d2[4])**2 + (y - d2[5])**2 + (z - d2[6])**2)
                    else:
                        theta = random.uniform(-np.pi, np.pi)
                        phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                        x = b_mon * np.cos(phi)*np.sin(theta) + properties[-2][4]
                        y = b_mon * np.sin(phi)*np.sin(theta) + properties[-2][5]
                        z = b_mon * np.cos(theta) + properties[-2][6]
                        props.append(x)
                        props.append(y)
                        props.append(z)

                        if EXTENDED == True and chain_pos > 3:
                            if properties[-3][2] != D[0]:
                                ind = -3
                            else:
                                ind = -4
                            d2 = properties[ind]
                            d0 = np.sqrt((x - d2[4])**2 + (y - d2[5])**2 + (z - d2[6])**2)
                            while (d0 < 0.5):
                                theta = random.uniform(-np.pi, np.pi)
                                phi = random.uniform(- 2 * np.pi, 2 * np.pi)
                                x = b_mon * np.cos(phi)*np.sin(theta) + properties[-1][4]
                                y = b_mon * np.sin(phi)*np.sin(theta) + properties[-1][5]
                                z = b_mon * np.cos(theta) + properties[-1][6]
                                props.append(x)
                                props.append(y)
                                props.append(z)
                                d0 = np.sqrt((x - d2[4])**2 + (y - d2[5])**2 + (z - d2[6])**2)                   
                properties.append(copy.deepcopy(props))
                mol_ang.append(atom_count)

                # normal bond - 1
                # regular bond - 1
                if chain_pos != (N[m_type]-1):
                    bonds.append([bond_count,1,atom_count,atom_count+1])
                    bond_count += 1
                atom_count += 1
        mol_angles.append(copy.deepcopy(mol_ang))
        mol_count += 1


#add salt
if SALT == True:
    tot_charge = 0
    for _ in range(N_cat):
        props = [atom_count,mol_count,Cation[0], q_cat]
        for xyz in range(dim):
            coord = np.random.uniform(0,1) * box_dim[xyz]
            props.append(coord)     
        properties.append(copy.deepcopy(props))
        atom_count += 1
        mol_count += 1
        tot_charge += q_cat
    for _ in range(N_anion):
        props = [atom_count,mol_count,Anion[0], q_cat]
        for xyz in range(dim):
            coord = np.random.uniform(0,1) * box_dim[xyz]
            props.append(coord)     
        properties.append(copy.deepcopy(props))
        atom_count += 1
        mol_count += 1
        tot_charge += q_anion
    print(f"Total charge: {tot_charge}")

if EXPLICIT_SOLVENT == True:
    for _ in range(N_sol):
        props = [atom_count,mol_count,W[0], q[0]/2]
        for xyz in range(dim):
            coord = np.random.uniform(0,1) * box_dim[xyz]
            props.append(coord)     
        properties.append(copy.deepcopy(props))

        bonds.append([bond_count,3,atom_count,atom_count+1])
        bond_count += 1
        atom_count += 1


        dx = np.random.choice([-1,1]) * b_mon/np.sqrt(3) + properties[-1][4]
        dy = np.random.choice([-1,1]) * b_mon/np.sqrt(3) + properties[-1][5]
        dz = np.random.choice([-1,1]) * b_mon/np.sqrt(3) + properties[-1][6]
        props = [atom_count,mol_count,D[0],-q[0]/2,dx,dy,dz]
        properties.append(copy.deepcopy(props))
        atom_count += 1
        mol_count += 1

if WLC == True:
    #process angles
    for mol in mol_angles:
        for i in range(len(mol)-2):
            angles.append([angle_count,1, mol[i], mol[i+1], mol[i+2]])
            angle_count += 1




# write output
with open(file_name + '.data', 'w') as fout:
    fout.writelines("First rule of programing: if it works then don't touch it!\n\n")
    fout.writelines(f'{atom_count - 1} atoms\n')
    fout.writelines(f'{bond_count - 1} bonds\n')
    fout.writelines(f'{angle_count - 1} angles\n')
    fout.writelines('\n')
    fout.writelines(f'{particle_types} atom types\n')
    fout.writelines(f'{bond_types} bond types\n')
    fout.writelines(f'{angle_types} angle types\n')
    fout.writelines('\n')
    fout.writelines(f'0.00000 {box_dim[0]} xlo xhi\n')
    fout.writelines(f'0.00000 {box_dim[1]} ylo yhi\n')
    fout.writelines(f'0.00000 {box_dim[2]} zlo zhi\n')
    fout.writelines('\n')
    fout.writelines('Masses\n')
    fout.writelines('\n')
    for i,mass in enumerate(masses):
        fout.writelines(f'{i + 1} {mass} \n')
    fout.writelines('\n')
    fout.writelines('Atoms\n')
    fout.writelines('\n')
    for line in properties:
        fout.writelines(f"{line[0]} {line[1]} {line[2]}  {line[3]}  {line[4]}  {line[5]}  {line[6]}\n")           
    fout.writelines('\n')
    fout.writelines('Bonds\n')
    fout.writelines('\n')
    for line in bonds:
        fout.writelines(f"{line[0]} {line[1]}  {line[2]} {line[3]}\n")
    fout.writelines('\n')
    fout.writelines('Angles\n')
    fout.writelines('\n')
    for line in angles:
        fout.writelines(f"{line[0]} {line[1]}  {line[2]} {line[3]} {line[4]}\n")

input = f"""Dim {dim}
max_steps {max_steps}
log_freq {log_freq}
traj_freq {traj_freq}
binary_freq {binary_freq}
charges {b_len} {smear_len}

pmeorder {1}

delt 0.005000
read_data input.data
compute avg_sk 1 freq 100 wait 0

integrator all GJF

Nx {Nx}
Ny {Ny}
Nz {Nz}

bond 1 harmonic 1.500000 0.0
bond 2 harmonic 135.000000 0.0
bond 3 harmonic 135.000000 0.0

angle 1 wlc 1.00000

n_gaussians 3
gaussian 1 1 3.000000 1.000000
gaussian 2 2 3.000000 1.000000
gaussian 1 2 3.000000 1.000000




"""
with open(file=file_name,mode='w') as fout:
    fout.writelines(input)
