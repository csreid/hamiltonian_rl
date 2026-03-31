sync:
	rsync -av --exclude 'post.py' --exclude='.venv' --exclude='__pycache__' --exclude='.git/' --exclude='*.pyc' . workstation:~/src/hamiltonian
