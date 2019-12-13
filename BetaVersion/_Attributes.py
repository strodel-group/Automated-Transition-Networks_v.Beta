class Attributes:

    """
    This class is used to group certain residues with specific features.
    Feel free to add own list of residues.
    """

    def __init__(self):
        self.hydrophobic = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'MET', 'TRP']
        self.polar = ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'TYR']
