cp /ccs/home/$USER/scratch/CD4/data/cd4.tar /mnt/bb/$USER
tar xf /mnt/bb/$USER/cd4.tar -C /mnt/bb/$USER/
mkdir /mnt/bb/$USER/trainimg
mkdir /mnt/bb/$USER/trainmsk
mkdir /mnt/bb/$USER/valimg
mkdir /mnt/bb/$USER/valmsk

mv /mnt/bb/$USER/train_e/originals /mnt/bb/$USER/trainimg/train
mv /mnt/bb/$USER/train_e/masks /mnt/bb/$USER/trainmsk/train
mv /mnt/bb/$USER/val_e/originals /mnt/bb/$USER/valimg/val
mv /mnt/bb/$USER/val_e/masks /mnt/bb/$USER/valmsk/val
