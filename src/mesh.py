import numpy as np


class Mesh:
    def __init__(self, left_boundary, right_boundary, n_nodes, n_ghosts):
        if left_boundary > right_boundary:
            raise ValueError('Left boundary of the mesh is greater than the right boundary.')
        self.n_ghosts = n_ghosts
        self.step = (right_boundary-left_boundary)/(n_nodes-1)
        self.nodes = np.linspace(left_boundary-n_ghosts*self.step,
                                right_boundary+n_ghosts*self.step,
                                n_nodes+2*n_ghosts)
        self.domain = range(n_ghosts,n_ghosts+n_nodes)
        self.left_ghosts = range(n_ghosts)
        self.right_ghosts = range(n_ghosts+n_nodes, n_nodes+2*n_ghosts)

    def __repr__(self):
        return f'This is a mesh with left ghost points {self.nodes[self.left_ghosts]}, '\
               f'domain {self.nodes[self.domain]}, and right ghost points {self.nodes[self.right_ghosts]}'


if __name__=='__main__':
    mesh = Mesh(0,10,11,2)
    assert np.allclose(mesh.nodes, np.arange(-2,13))
    assert np.allclose(mesh.nodes[mesh.domain], np.arange(0,11))
    assert np.allclose(mesh.nodes[mesh.left_ghosts], np.arange(-2,0))
    assert np.allclose(mesh.nodes[mesh.right_ghosts], np.arange(11,13))
    print(mesh)
    large_mesh = Mesh(-50.0,0.,1001,2)
    print(large_mesh)


