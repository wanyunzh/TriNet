# TriNet
## 1.Making predictions for sequences

### 1.1 Description

TriNet is a tri-fusion netural network for anticancer and antimicrobial peptides recognition. By inputting the peptide sequences and PSSM profiles, this software can use the pre-trained model to identify anticancer and antimicrobial peptides.

### 1.2 Getting PSSM profiles

you can get your PSSM files through standalone BLAST program.

* **Step** 1. Go to the NCBI website to download the BLAST program

  https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/. <br />

  Go to website https://ftp.ncbi.nlm.nih.gov/blast/db/ to download protein sequences.<br />

  Put swissprot.tar.gz to the bin folder (eg. ./blast/bin/).

* **Step 2.** Use command line to enter the /blast/bin/ folder, run the following command to establish a local BLAST database.

```

makeblastdb.exe -in swissprot -dbtype prot -title “swissprot” -out lxsp

```

* **Step 3.** Put a single sequence into the queryseq.fasta file, run the following command to generate a single PSSM such as 1.pssm. and rename them to TXT files such as 1.txt
```
psiblast.exe -db lxsp -query 1.txt -evalue 0.001 -num_iterations 3 -out_ascii_pssm 1.pssm
```

* **Note**. There are some requirments for PSSM files:

  1.the PSSM files can only be named using numbers and it should be TXT files. For example, 1.txt, 2.txt ...

  2.the name of the PSSM files must match the sequence number of your predicting sequences in case some sequences may not have PSSM files.

  3.All PSSM files should be put into a folder.

  (These requirements can be done simply via the aboved mention instruction )

### 1.3 Requirements for prediction

TriNet can be used under Linux or Windows environment.

Python 3.9.7

Python packages: Tensorfollow(vr.2.8.0), pandas and scipy

### 1.4 Starting a prediction

* **prediction of anticancer or antimicrobial peptides**

usage: TriNet.py [-h] [--PSSM_file PSSM_FILE] [--sequence_file  SEQUENCE_FILE] [--output OUTPUT] [--operation_mode OPERATION_MODE]

* **Rquired**

--PSSM_file PSSM_FILE, -p PSSM_FILE

path of PSSM  files

--sequence_file SEQUENCE_FILE, -s SEQUENCE_FILE

path of sequence file

--operation_mode OPERATION_MODE, -mode OPERATION_MODE

c for anticancer prediction and m for antimicrobial prediction

* **Optional**

-h, --help show this help message and exit

--output OUTPUT, -o OUTPUT

path of Trinet result,  defaut with path of current path/output

* **Note**

1.There are four options for '-mode' item, they are sc,sm,fc and fm. s is standard mode(need users to provide pssm) f is fast mode(only need to provide fasta files) c for anticancer peptides prediction and m for antimicrobial peptides prediction.

2.If the '-o' output item is empty in your command line, the corresponding result file will be placed in the current  working path.

* **Typical commands**

The following command is an example for anticancer peptide prediction:

```

$ python ./TriNet/TriNet.py -mode sc -s ./TriNet/ACP_example.fasta -p ./TriNet/pssm_acp_example/ -o ./TriNet/acpout.csv

```

## 2.Repeating experiments in the paper.

All codes and data are placed in the TriNet-Reproducing.zip file. This file needs to be decompressed before reproducing the results and you can read README.md in TriNet-Reproducing.zip for more information.

## Contact

Any questions, problems, bugs are welcome and should be dumped to

wanyunzh@gmail.com
