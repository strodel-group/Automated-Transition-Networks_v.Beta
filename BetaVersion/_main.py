"Please note that parts of the script 'element.py' contained in mdtraj were used to manage the elements."

import h5py as hdf5
import itertools
import mdtraj
import networkx as nx
import numpy as np
import os
from ._PSE import get_by_symbol
from ._Attributes import Attributes
import sys

class General:
    def __init__(self, Topology=None, Trajectory=None, Number_of_Molecules=0, Index_Array=None, Elements_per_Chain=0):
        self.top = Topology
        self.traj = Trajectory
        self.n_mol = Number_of_Molecules

        try:
            if (isinstance(Index_Array, int) == True):
                self.idx_arr = np.arange(0, Index_Array)
                self.len_idx_arr = Index_Array

            else:
                self.idx_arr = Index_Array
                self.len_idx_arr = len(Index_Array)

        except TypeError:
            raise TypeError('Index_Array must be 1-dimensional array or integer.')


        try:
            if (Elements_per_Chain != 0):
                self.n_epc = Elements_per_Chain
            else:
                self.n_epc = self.len_idx_arr

        except TypeError:
            raise TypeError('Elements_per_Chain must be integer.')

    def _Correlate_Index_And_Chain(self):
        """
        Returns a two-dimensional array (number_of_molecules, indices).
        """
        index_chain_array = np.zeros((self.n_mol, self.len_idx_arr), dtype=int)

        for chain in range(self.n_mol):
            for idx, value in enumerate(self.idx_arr):
                index_chain_array[chain][idx] = value + self.n_epc*chain

        return (index_chain_array)

    def _Index_Belongs_To_Chain(self, Index=0):
        """
        Function which returns the number of chain belonging to given index.
        """
        idx_arr = self._Correlate_Index_And_Chain()

        return (np.where(Index == idx_arr)[0][0])

    def _Residues_Belong_To_Attribute(self, Attribute_Array):
        index_list = []

        Residues = self._Correlate_Index_And_Chain()[0]
        for resid in Residues:
            if (str(self.top.residue(resid))[:3] in Attribute_Array):
                index_list.append(resid)

        return (index_list)

    def _Determine_Intermolecular_Pairs(self):
        Intermolecular_Pairs = {}
        arr = np.array(self.idx_arr)
        for idx1 in range(self.n_mol):
            for idx2 in range(idx1+1, self.n_mol):

                chain1 = idx1*self.n_epc + arr
                chain2 = idx2*self.n_epc + arr

                indices_in_idx1_idx2 = list(itertools.product(chain1, chain2))
                Intermolecular_Pairs.update({str([idx1, idx2]): indices_in_idx1_idx2})

        return (Intermolecular_Pairs)

    def _Compute_Contacts_Between_Residues(self, Pairs=None, Cutoff=0.5, Trajectory=None):

            output = mdtraj.compute_contacts(Trajectory, Pairs)

            distances = output[0].T
            atom_pairs = output[1]

            residues_in_contact = []
            contacts = []
            idx = 0
            cont = 0
            for frame in distances:
                for d in frame:
                    if (d < Cutoff):
                        atom_pair = atom_pairs[idx]

                        resid1 = self.top.residue(atom_pair[0])
                        resid2 = self.top.residue(atom_pair[1])

                        arr = [resid1, resid2]

                        if (arr not in residues_in_contact):
                            residues_in_contact.append(arr)

                    cont += 1

                contacts.append(cont)
                cont = 0
                idx += 1

            return (residues_in_contact, contacts)

    def _Intermolecular_Salt_Bridge_Pairs(self, Charge_Per_Molecule=False):
        table, _ = self.top.to_dataframe()

        elements = [element for element in table["element"]]
        atoms = [atom for atom in self.top.atoms]

        salt_bridge_acceptor = []
        pos = 0
        for N_index in self.top.select("symbol == N"):

            bonding_degree = 0
            for bond in self.top.bonds:
                if (atoms[N_index] in bond):
                    bonding_degree += 1

            if (bonding_degree == 4):
                salt_bridge_acceptor.append(N_index)
                pos += 1

        salt_bridge_donor = []
        neg = 0
        for O_index in self.top.select('symbol == O and is_sidechain'):

            try:
                bonding_degree = 0
                for bond in self.top.bonds:
                    if (atoms[O_index] in bond):
                        if (elements[O_index + 1] == 'O'):
                            bonding_degree += 1

                if (bonding_degree == 1):
                    salt_bridge_donor.append(O_index)
                    salt_bridge_donor.append(O_index + 1)
                    neg -= 1

            except IndexError:
                pass    # OXT

        charge_per_molecule = (pos + neg)/self.n_mol

        atoms_per_mol = int(self.top.n_atoms/self.n_mol)

        salt_bridges = {}
        for salt_bridge_donor_index in salt_bridge_donor:
            salt_bridge_donor_chain = int(salt_bridge_donor_index/atoms_per_mol)

            for salt_bridge_acceptor_index in salt_bridge_acceptor:
                salt_bridge_acceptor_chain =  int(salt_bridge_acceptor_index/atoms_per_mol)

                if (salt_bridge_acceptor_chain != salt_bridge_donor_chain):
                    try:
                        temp = salt_bridges[str([salt_bridge_donor_chain, salt_bridge_acceptor_chain])]
                        temp.append([salt_bridge_donor_index, salt_bridge_acceptor_index])
                        salt_bridges.update({str([salt_bridge_donor_chain, salt_bridge_acceptor_chain]): temp})

                    except KeyError:
                        salt_bridges.update({str([salt_bridge_donor_chain, salt_bridge_acceptor_chain]): [[salt_bridge_donor_index, salt_bridge_acceptor_index]]})

        if (Charge_Per_Molecule == True):
            print('Charge_Per_Molecule:', charge_per_molecule)

        return(salt_bridges)

class Features:

    def __init__(self, Trajectory=None, Topology=None, Number_of_Chains=0, Chunk_Index=0, Chunksize=0):
        self.traj = Trajectory
        self.n_fram = Trajectory.n_frames

        self.top = Topology
        self.n_atoms = Topology.n_atoms
        self.n_res = Topology.n_residues

        self.n_mol = Number_of_Chains

        """
        Check if chains are of equal length.
        """
        if (self.n_atoms%self.n_mol != 0):
            raise ValueError('The given chains are not of equal length.')
        else:
            self.atoms_per_mol = int(self.n_atoms/self.n_mol)

        if (self.n_res%self.n_mol != 0):
            raise ValueError('The given chains are not of equal length.')
        else:
            self.res_per_mol = int(self.n_res/self.n_mol)

        """
        Chunk index and chunksize to correct frame number. Compute trajectories start always from frame 0.
        """
        self.chunk_idx = Chunk_Index
        self.chunksize = Chunksize

    """
    Functions for oligomersize feature.
    """

    def _Calculate_Oligomersize(self, Cutoff=0.5):

        g = General(self.top, self.traj, self.n_mol, self.atoms_per_mol)
        indices_and_chain = g._Correlate_Index_And_Chain()

        for number_of_frame, frame in enumerate(self.traj):
            Set = [set([s]) for s in range(self.n_mol)]
            for chain in range(self.n_mol):
                atoms_in_contact = mdtraj.compute_neighbors(frame, Cutoff, indices_and_chain[chain])
                for atom_index in atoms_in_contact[0]:
                    if (atom_index not in indices_and_chain[chain]):
                        chain1 = g._Index_Belongs_To_Chain(atom_index)
                        Set[chain1].add(chain)
                        Set[chain].add(chain1)

            for i in range(self.n_mol):
                for j in range(self.n_mol):
                    if (i < j):
                        if (len(list(Set[i] & Set[j])) != 0):
                            united = Set[i].union(Set[j])
                            Set[i] = united
                            Set[j] = united

            largest_oligomer = max([len(Set[s]) for s in range(self.n_mol)])

            """
            Correct the frame number and write results in *.hdf5 file.
            """
            frame_number_corr = number_of_frame + self.chunk_idx*self.chunksize

            """
            hdf5 does not support sets -> convert to list.
            """
            Set = [list(s) for s in Set]

            """
            Create a dataset for every frame, because arrays have different lengths.
            """

            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'a') as f:
                grp = f.require_group("Oligomers")
                sbgrp = grp.require_group(str(frame_number_corr))
                for chain, s in enumerate(Set):
                    sbgrp.create_dataset(str(chain), data=s)

                f["Largest_Oligomer"][frame_number_corr] = largest_oligomer

        return ()

    """
    Functions for compactness feature.
    """

    def _Get_Mass_Of_Topology_Atoms(self):

        mass_of_topology = np.zeros(self.n_atoms, dtype=float)
        table, _ = self.top.to_dataframe()

        for atom_index, element in enumerate(table["element"]):
            mass_of_topology[atom_index] = get_by_symbol(element).mass
        splitted_mass_of_topology = np.split(mass_of_topology, self.n_mol)

        return (splitted_mass_of_topology[0])

    def _Split_XYZ_Coordinates(self, Trajectory_Frame, Structures_In_Frame):

        """
        This function produces the xyz coordinates array which is needed as input for
        computing the moment of inertia tensor. The unique_structures array is needed
        for relating the coordinates to the oligomers.
        """

        splitted_xyz = np.split(Trajectory_Frame.xyz, self.n_mol, axis=1)

        """
        Get unique structures in frame:
        """
        unique_structures = []
        for structure in Structures_In_Frame:
            if (structure not in unique_structures):
                unique_structures.append(structure)

        """
        Extract coordinates of unique structures.
        """

        unique_xyz = []
        for struct in unique_structures:

            oligomer_xyz = []
            for chain in struct:
                oligomer_xyz.extend(splitted_xyz[chain][0])
            unique_xyz.append(oligomer_xyz)

        return (unique_xyz, unique_structures)

    def _Moment_Of_Intertia_Ratio(self, XYZ_Coordinates, Masses):

        moment_of_inertia_ratio_array = []
        for xyz in XYZ_Coordinates:

            """
            Since Masses-array contains only atoms of one chain, it has to be extended
            for oligomers. This is done by using np.tile().
            """
            fraction = int(np.shape(xyz)[0]/np.shape(Masses)[0])
            if (fraction > 1):
                masses = np.tile(Masses, fraction)
            else:
                masses = Masses

            """
            Coordinates are transformed into the center of mass system.
            """
            center_of_mass = np.average(xyz, axis=0, weights=masses)
            coordinates_relative_to_center_of_mass = np.subtract(xyz, center_of_mass)

            x = coordinates_relative_to_center_of_mass[:,0]
            y = coordinates_relative_to_center_of_mass[:,1]
            z = coordinates_relative_to_center_of_mass[:,2]

            """
            Elements of moment of inertia tensor are computed.
            """
            moment_of_inertia_tensor = np.zeros((3, 3), dtype=float)

            moment_of_inertia_tensor[0][0] = sum(masses*(np.power(y, 2) + np.power(z, 2)))    # I_xx
            moment_of_inertia_tensor[1][1] = sum(masses*(np.power(x, 2) + np.power(z, 2)))    # I_yy
            moment_of_inertia_tensor[2][2] = sum(masses*(np.power(x, 2) + np.power(y, 2)))    # I_zz

            moment_of_inertia_tensor[0][1] = - sum(masses*x*y)      # I_xy
            moment_of_inertia_tensor[1][0] = moment_of_inertia_tensor[0][1]     # I_yx

            moment_of_inertia_tensor[0][2] = - sum(masses*x*z)      # I_xz
            moment_of_inertia_tensor[2][0] = moment_of_inertia_tensor[0][2]     # I_zx

            moment_of_inertia_tensor[1][2] = - sum(masses*y*z)      # I_yz
            moment_of_inertia_tensor[2][1] = moment_of_inertia_tensor[1][2]     # I_zy

            """
            The eigenvalues of the moment of inertia tensor correspond to the moments
            of inertia on the principle axes.
            """
            Lambda, _ = np.linalg.eig(moment_of_inertia_tensor)
            Lambda = np.sort(Lambda)

            """
            As feature the ration of highest and lowest main moment of inertia is used.
            See:

            Pathways of Amyloid-β Aggregation Depend on Oligomer Shape
            B. Barz, Q. Liao, B. Strodel
            J. Am. Chem. Soc., 140: 319-327 (2018)

            Ratio in [1, 10] with 1 = linear shape, 10 = spherical, compact shape.
            """

            moment_of_inertia_ratio_array.append(int(abs(round(10*Lambda[0]/Lambda[2]))))

        return (moment_of_inertia_ratio_array)

    def _Compactness(self, Oligomers_List):

        mass = self._Get_Mass_Of_Topology_Atoms()

        """
        Compactness Array
        """
        compactness = np.zeros((self.n_fram, self.n_mol), dtype=int)
        for (number_of_frame, frame), structures in zip(enumerate(self.traj), Oligomers_List):
            un_xyz, un_struct  = self._Split_XYZ_Coordinates(frame, structures)

            ratio = self._Moment_Of_Intertia_Ratio(un_xyz, mass)

            for idx, val in zip(un_struct, ratio):
                for chain in idx:
                    compactness[number_of_frame][chain] = val

        return (compactness)

    """
    Functions for hydrophobicity feature.
    """

    def _Frames_With_Oligomers(self, Largest_Oligomer):

        frames_with_oligomers_list_global = []
        for number_of_frame, oligomersize in enumerate(Largest_Oligomer):
            if (oligomersize > 1):

                frame_number_corr = number_of_frame + self.chunk_idx*self.chunksize
                frames_with_oligomers_list_global.append(frame_number_corr)

        return (frames_with_oligomers_list_global)

    def _Hydrophobic_Contacts_Dictionary(self, Frames_With_Oligomers=None, Oligomers_In_Frame=None):

        att = Attributes()

        resid = General(Topology=self.top, Trajectory=self.traj, Number_of_Molecules=self.n_mol, Index_Array=self.res_per_mol)
        hydrophobic_residues = resid._Residues_Belong_To_Attribute(Attribute_Array=att.hydrophobic)

        resid_hydr = General(Topology=self.top, Trajectory=self.traj, Number_of_Molecules=self.n_mol, Index_Array=hydrophobic_residues, Elements_per_Chain=self.res_per_mol)
        hydr_pairs = resid_hydr._Determine_Intermolecular_Pairs()

        return (resid_hydr, hydr_pairs)

    """
    Functions for saltbridge contact feature.
    """

class Reader:

    def __init__ (self, Number_of_Molecules, Number_of_Frames, Filename='Features.hdf5'):
        self.file = Filename
        self.n_mol = Number_of_Molecules
        self.n_fram = Number_of_Frames

    def Extract_States(self, *args):
        states = []
        number_of_features = len(args)
        for frame, args in enumerate(zip(*args)):
            states_in_frame = []
            for mol in range(self.n_mol):
                states_in_frame.append([args[arg][mol] for arg in range(number_of_features)])

            states.append(states_in_frame)

        return (states)

    def Transition_Matrix(self, States, Start_Frame=0, End_Frame=None):
        if (End_Frame == None):
            End_Frame = self.n_fram

        different_states_list = []
        population_of_states_dict = {}
        transition_matrix_dict = {}
        idx = 0
        for frame, states in enumerate(States[Start_Frame:End_Frame]):
            for state in states:
                if (state not in different_states_list):
                    different_states_list.append(state)
                    population_of_states_dict.update({tuple(state): 1})
                    transition_matrix_dict.update({tuple(state): idx})
                    idx += 1

                else:
                    population = population_of_states_dict[tuple(state)]
                    population += 1
                    population_of_states_dict.update({tuple(state): population})
        different_states = len(different_states_list)

        transition_matrix = np.zeros((different_states, different_states), dtype=int)
        States = np.array(States)
        for mol in range(self.n_mol):
            state_history = States[Start_Frame:End_Frame, mol, :]
            for state1, state2 in zip(state_history[:-1], state_history[1:]):
                idx1 = transition_matrix_dict[tuple(state1)]
                idx2 = transition_matrix_dict[tuple(state2)]

                transition_matrix[idx1][idx2] += 1

        return (transition_matrix, transition_matrix_dict)


    def Network_Data(self, *args,  Min_Population=0.0, Gexf_Name="Network.gexf", **kwargs):
        states = self.Extract_States(*args)
        transition_matrix, transition_matrix_dict = self.Transition_Matrix(states, **kwargs)

        if (Min_Population == 0.0):
            nodes_dict = {}
            for state, size in zip(transition_matrix_dict.keys(), np.diagonal(transition_matrix)):
                nodes_dict.update({state: size})

            edges_dict = {}
            for state1, (idx1, row) in zip(transition_matrix_dict.keys(), enumerate(transition_matrix)):
                for state2, (idx2, element) in zip(transition_matrix_dict.keys(), enumerate(row)):
                    if (idx1 != idx2 and element != 0):
                        edges_dict.update({(state1, state2): element})

        else:
            max_population = max(np.diag(transition_matrix))
            nodes_dict = {}
            for state, size in zip(transition_matrix_dict.keys(), np.diagonal(transition_matrix)):
                fraction = size/max_population
                if (fraction >= Min_Population):
                    nodes_dict.update({state: size})

            edges_dict = {}
            for state1, (idx1, row) in zip(transition_matrix_dict.keys(), enumerate(transition_matrix)):
                for state2, (idx2, element) in zip(transition_matrix_dict.keys(), enumerate(row)):
                    if (idx1 != idx2 and element != 0):
                        if (state1 in nodes_dict.keys() and state2 in nodes_dict.keys()):
                            edges_dict.update({(state1, state2): element})

        G = nx.DiGraph()
        for k, v in nodes_dict.items():
            G.add_node(k, size=float(v))
        for k, v in edges_dict.items():
            G.add_edge(k[0], k[1], weight=float(v))
	nx.draw(G)
        nx.write_gexf(G, Gexf_Name)

        return(nodes_dict, edges_dict)

    def Oligomers(self, Start_Frame=0, End_Frame=None, Speak=False):
        if (End_Frame == None):
            End_Frame = self.n_fram

        Oligomers_List = []
        if (Speak == True):
            print('Frame \t Oligomer structures')
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                for fram in range(Start_Frame, End_Frame):
                    Oligomers_List_Frame = [list(f["Oligomers"][str(fram)][str(chain)][:])\
                     for chain in range(self.n_mol)]
                    print(fram, '\t', Oligomers_List_Frame)
                    Oligomers_List.append(Oligomers_List_Frame)

        else:
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                for fram in range(Start_Frame, End_Frame):
                    Oligomers_List.append([list(f["Oligomers"][str(fram)][str(chain)][:])\
                     for chain in range(self.n_mol)])

        return (Oligomers_List)

    def Oligomersize(self, Start_Frame=0, End_Frame=None, Speak=False):
        if (End_Frame == None):
            End_Frame = self.n_fram

        Oligomersize_List = []
        if (Speak == True):
            print('Frame \t Oligomer structures')
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                for fram in range(Start_Frame, End_Frame):
                    Oligomersize_List_Frame = [list(len(f["Oligomers"][str(fram)][str(chain)][:]))\
                     for chain in range(self.n_mol)]
                    print(fram, '\t', Oligomersize_List_Frame)
                    Oligomersize_List.append(Oligomersize_List_Frame)

        else:
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                for fram in range(Start_Frame, End_Frame):
                    Oligomersize_List.append([len(list(f["Oligomers"][str(fram)][str(chain)][:]))\
                     for chain in range(self.n_mol)])

        return (Oligomersize_List)



    def Largest_Oligomer(self, Start_Frame=0, End_Frame=None, Speak=False):
        if (End_Frame == None):
            End_Frame = self.n_fram

        Largest_Oligomer_List = []
        if (Speak == True):
            print('Frame \t Largest_Oligomer')
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                for fram in range(Start_Frame, End_Frame):
                    value = f["Largest_Oligomer"][fram]
                    print(fram, '\t', value)
                    Largest_Oligomer_List.append(value)
        else:
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                Largest_Oligomer_List = [f["Largest_Oligomer"][fram] for fram in range(Start_Frame, End_Frame)]

        return (Largest_Oligomer_List)

    def Compactness(self, Start_Frame=0, End_Frame=None, Speak=False):
        if (End_Frame == None):
            End_Frame = self.n_fram

        if (Speak == True):
            print('Frame \t Compactness')
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                [print(fram, '\t', f["Compactness"][str(fram)][:]) for fram in range(Start_Frame, End_Frame)]

        Compactness_List = []
        with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
            Compactness_List = [f["Compactness"][str(fram)][:] for fram in range(Start_Frame, End_Frame)]

        return (Compactness_List)

    def Hydrophobic_Contacts(self, Start_Frame=0, End_Frame=None, Speak=False):
        if (End_Frame == None):
            End_Frame = self.n_fram

        if (Speak == True):
            print('Frame \t Chain \t Hydrophobic Contacts')
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                for fram in range(Start_Frame, End_Frame):
                    """
                    Since not all frames include oligomers, KeyErrors have to be suppressed.
                    """
                    try:
                        group = f["Hydrophobic_Contacts"][str(fram)]
                        for k in group.keys():
                            print(fram, '\t', k, '\t', group[k][0])
                    except KeyError:
                        pass

        Hydrophobic_Contacts_List = np.zeros((self.n_fram, self.n_mol), dtype=int)
        with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
            for fram in range(Start_Frame, End_Frame):
                """
                Since not all frames include oligomers, KeyErrors have to be suppressed.
                """
                try:
                    group = f["Hydrophobic_Contacts"][str(fram)]
                    for k in group.keys():
                        Hydrophobic_Contacts_List[fram][int(k)] = group[k][0]
                except KeyError:
                    pass

        return (Hydrophobic_Contacts_List)

    def Saltbridge_Contacts(self, Start_Frame=0, End_Frame=None, Speak=False):
        if (End_Frame == None):
            End_Frame = self.n_fram

        if (Speak == True):
            print('Frame \t Chain \t Saltbridge Contacts')
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
                for fram in range(Start_Frame, End_Frame):
                    """
                    Since not all frames include oligomers, KeyErrors have to be suppressed.
                    """
                    try:
                        group = f["Saltbridge_Contacts"][str(fram)]
                        for k in group.keys():
                            print(fram, '\t', k, '\t', group[k][0])
                    except KeyError:
                        pass

        Saltbridge_Contacts_List = np.zeros((self.n_fram, self.n_mol), dtype=int)
        with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'r') as f:
            for fram in range(Start_Frame, End_Frame):
                """
                Since not all frames include oligomers, KeyErrors have to be suppressed.
                """
                try:
                    group = f["Saltbridge_Contacts"][str(fram)]
                    for k in group.keys():
                        Saltbridge_Contacts_List[fram][int(k)] = group[k][0]
                except KeyError:
                    pass

        return (Saltbridge_Contacts_List)

class Compute(Reader):

    def __init__(self, Trajectory=None, Topology=None, Number_of_Molecules=1, Number_of_Frames=0, Chunksize=None, Write_hdf5=True):
        self.traj = Trajectory
        self.top = mdtraj.load(Topology).topology
        self.n_mol = Number_of_Molecules
        self.n_fram = Number_of_Frames

        if (Chunksize == None):
            self.chunksize = self.n_fram
        else:
            self.chunksize = Chunksize


        if (Write_hdf5 == True):
            """
            Create a new *.hdf5 file.
            """
            with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'w') as f:
                pass

        Reader.__init__(self, Number_of_Molecules, Number_of_Frames)

    def Oligomersize(self, Cutoff=0.5):
        """
        Default cutoff = 0.5 nm.
        For every feature that is chosen, a dataset has to be created in *.hdf5 file.
        """
        with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'a') as f:
            f.create_group("Oligomers")
            f.create_dataset("Largest_Oligomer", data=np.ones(self.n_fram, dtype=int))

        for chunk_idx, chunk_traj in enumerate(mdtraj.iterload(self.traj, top=self.top, chunk=self.chunksize)):
            feat = Features(chunk_traj, self.top, self.n_mol, chunk_idx, self.chunksize)
            feat._Calculate_Oligomersize(Cutoff)

        return ()

    def Saltbridge_Contacts(self, Cutoff=1.5, Overwrite=False):
        """
        Check if dataset 'Saltbridge_Contacts' already exists. If so, ask for overwriting, default=True.
        """

        with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'a') as f:
            if (Overwrite == False):
                try:
                    f.create_group("Saltbridge_Contacts")
                except ValueError:
                    print("ValueError: Group 'Saltbridge_Contacts' already exists in 'Features.hdf5'. If you want to overwrite it give 'Overwrite=True' as input. \n",
                    f["Saltbridge_Contacts"])
                    sys.exit()
            else:
                del f["Saltbridge_Contacts"]
                f.create_group("Saltbridge_Contacts")


        """
        Compute 'Saltbridge_Contacts' feature.
        """
        atoms_per_mol = int(self.top.n_atoms/self.n_mol)

        g = General(Topology=self.top, Number_of_Molecules=self.n_mol, Index_Array=atoms_per_mol)
        salt_bridges_dict = g._Intermolecular_Salt_Bridge_Pairs()

        """
        Check if residues are charged and salt bridges possible.
        """
        if (len(salt_bridges_dict.keys()) == 0):
            return (print('Molecule has no charged residues. No salt bridges possible.'))

        else:
            for chunk_idx, chunk_traj in enumerate(mdtraj.iterload(self.traj, top=self.top, chunk=self.chunksize)):

                size = len(chunk_traj)
                start = chunk_idx*self.chunksize
                end = chunk_idx*self.chunksize + size

                largest_oligomer_list = super().Largest_Oligomer(Start_Frame=start, End_Frame=end)

                feat = Features(chunk_traj, self.top, self.n_mol, chunk_idx, self.chunksize)
                frames_in_chunk_with_oligomers = feat._Frames_With_Oligomers(largest_oligomer_list)

                oligomers_in_frame = []
                for frame in feat._Frames_With_Oligomers(largest_oligomer_list):

                    """
                    Get unique structures in frame:
                    """
                    unique_structures = []
                    for structure in super().Oligomers(Start_Frame=frame, End_Frame=frame+1)[0]:
                        if (structure not in unique_structures):
                            unique_structures.append(structure)

                    """
                    Get oligomers in frame:
                    """
                    oligomers_in_this_frame = []
                    for structure in unique_structures:
                        if (len(structure) > 1):
                            oligomers_in_this_frame.append(structure)
                    oligomers_in_frame.append(oligomers_in_this_frame)


                for frame, oligomers in zip(frames_in_chunk_with_oligomers, oligomers_in_frame):
                    for oligomer in oligomers:
                        frame_loc = frame - start
                        salt_bridge_distance_matrix = mdtraj.compute_distances(chunk_traj[frame_loc], salt_bridges_dict[str(oligomer)])

                        contact = len(list(itertools.filterfalse(lambda x: x > Cutoff, salt_bridge_distance_matrix[0])))

                        with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'a') as f:
                            grp = f.require_group("Saltbridge_Contacts")
                            sub_grp = grp.require_group(str(frame))
                            for monomer in oligomer:
                                sub_grp.create_dataset(str(monomer), data=(1, ), dtype=int)
                                f["Saltbridge_Contacts"][str(frame)][str(monomer)][0] = contact

            return ()

    def Hydrophobic_Contacts(self, Cutoff=1.5, Overwrite=False):

        """
        Check if dataset 'Hydrophobic_Contacts' already exists. If so, ask for overwriting, default=True.
        """

        with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'a') as f:
            if (Overwrite == False):
                try:
                    f.create_group("Hydrophobic_Contacts")
                except ValueError:
                    print("ValueError: Group 'Hydrophobic_Contacts' already exists in 'Features.hdf5'. If you want to overwrite it give 'Overwrite=True' as input. \n",
                    f["Hydrophobic_Contacts"])
                    sys.exit()
            else:
                del f["Hydrophobic_Contacts"]
                f.create_group("Hydrophobic_Contacts")


        """
        Compute 'Hydrophobic_Contacts' feature.
        """

        for chunk_idx, chunk_traj in enumerate(mdtraj.iterload(self.traj, top=self.top, chunk=self.chunksize)):

            size = len(chunk_traj)
            start = chunk_idx*self.chunksize
            end = chunk_idx*self.chunksize + size

            largest_oligomer_list = super().Largest_Oligomer(Start_Frame=start, End_Frame=end)

            feat = Features(chunk_traj, self.top, self.n_mol, chunk_idx, self.chunksize)
            frames_in_chunk_with_oligomers = feat._Frames_With_Oligomers(largest_oligomer_list)

            oligomers_in_frame = []
            for frame in feat._Frames_With_Oligomers(largest_oligomer_list):

                """
                Get unique structures in frame:
                """
                unique_structures = []
                for structure in super().Oligomers(Start_Frame=frame, End_Frame=frame+1)[0]:
                    if (structure not in unique_structures):
                        unique_structures.append(structure)

                """
                Get oligomers in frame:
                """
                oligomers_in_this_frame = []
                for structure in unique_structures:
                    if (len(structure) > 1):
                        oligomers_in_this_frame.append(structure)
                oligomers_in_frame.append(oligomers_in_this_frame)

            resid_hydr, hydr_pairs = feat._Hydrophobic_Contacts_Dictionary(Frames_With_Oligomers=frames_in_chunk_with_oligomers, Oligomers_In_Frame=oligomers_in_frame)

            for frame, oligomer in zip(frames_in_chunk_with_oligomers, oligomers_in_frame):
                with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'a') as f:
                    grp = f.require_group("Hydrophobic_Contacts")
                    sub_grp = grp.create_group(str(frame))
                    for chains in oligomer:
                        residues_in_contact = []
                        for idx1, chain1 in enumerate(chains):
                            for chain2 in chains[idx1+1:]:
                                pair = str([chain1, chain2])

                                residues_in_contact_chain, _ = resid_hydr._Compute_Contacts_Between_Residues(hydr_pairs[pair], Cutoff=Cutoff, Trajectory=chunk_traj)
                                residues_in_contact.extend(residues_in_contact_chain)

                        for monomer in chains:
                            sub_grp.create_dataset(str(monomer), data=(1,), dtype=int)
                            f["Hydrophobic_Contacts"][str(frame)][str(monomer)][0] = len(residues_in_contact)

        return ()

    def Compactness(self, Overwrite=False):

        """
        Check if dataset 'Compactness' already exists. If so, ask for overwriting, default=True.
        """
        with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'a') as f:
            if (Overwrite == False):
                try:
                    f.create_group("Compactness")
                except ValueError:
                    print("ValueError: Group 'Compactness' already exists in 'Features.hdf5'. If you want to overwrite it give 'Overwrite=True' as input. \n",
                     f["Compactness"])
                    sys.exit()
            else:
                del f["Compactness"]
                f.create_group("Compactness")


        """
        Compute 'Compactness' feature by creating 'oligomers_list' and moment of inertia tensor chunkwise.
        """
        for chunk_idx, chunk_traj in enumerate(mdtraj.iterload(self.traj, top=self.top, chunk=self.chunksize)):

            size = len(chunk_traj)
            start = chunk_idx*self.chunksize
            end = chunk_idx*self.chunksize + size

            oligomers_list = super().Oligomers(Start_Frame=start, End_Frame=end)

            feat = Features(chunk_traj, self.top, self.n_mol, chunk_idx, self.chunksize)

            """
            Create a dataset for every frame, because arrays have different lengths.
            """

            for frame, comp in enumerate(feat._Compactness(oligomers_list)):
                frame_number_corr = frame + start

                with hdf5.File(os.path.join(os.getcwd(), 'Features.hdf5'), 'a') as f:
                    grp = f.require_group("Compactness")
                    grp.create_dataset(str(frame_number_corr), data=comp)

        return ()
