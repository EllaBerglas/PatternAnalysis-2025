# Conda environment
    conda create --name COMP3710_Project python=3.10
    conda activate COMP3710_Project
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia


# Models
https://arxiv.org/pdf/2107.00645
https://arxiv.org/pdf/2201.03545 


# rangpur commands
ssh s4742538@rangpur.compute.eait.uq.edu.au

Copy a file from remote to local:
scp s4742538@rangpur.compute.eait.uq.edu.au:/alzhiemers_data.zip .

Copy a file from local to remote:
scp C:\local\path\file.txt username@hostname:/remote/path/



# Task
Classify Alzheimer’s disease (normal and AD) of the ADNI brain data (see Appendix for link) using one
of the latest vision models such as the ConvNeXt [9] or GFNet [10] set having a minimum accuracy of 0.8
on the test set. [Hard Difficulty]

Original repository: https://github.com/shakes76/PatternAnalysis-2025/ 
Forked Rpository: https://github.com/EllaBerglas/PatternAnalysis-2025/tree/topic-recognition

# Marking Scheme Recognition (20 marks)
1. Algorithm solves the respective pattern recognition problem appropriately (5 Marks)
2. Algorithm implemented in TF/PyTorch demonstrated and functions as intended to solve the respective
problem (3 Marks)
3. Good design implemented in TF/PyTorch that solves the problem (1 Marks)
4. Commenting (1 Mark)
5. Algorithm is normal or above difficulty while implemented in TF/PyTorch while solving the respective
problem (5 Marks)
6. Algorithm is hard difficulty while implemented in TF/PyTorch while solving the respective problem (5
Marks)

# Marking Scheme Commit Log (5 Marks, Pass Hurdle)
1. Meaningful commit messages with evidence of individual work and originality (2 Marks)
2. 2. Progressive commits and logical log structure providing evidence of individual work and originality (3 Marks)

# Marking Criteria Documentation (10 Marks) (code comments and readme)
1. description and explanation of the working principles of the algorithm implemented and the problem it solves (5 Marks)
2. description of usage and comments throughout scripts (3 Marks)
3. proper formatting using GitHub markdown (2 Mark)

# Pull Request Marking Criteria (5 marks)
1. Creating a pull-request into correct branch with a working and demonstrable version of the algorithm (2 Marks)
2. Incorporating feedback into the pull request (2 Marks)
3. Description of and comments within the pull request (1 Mark)
