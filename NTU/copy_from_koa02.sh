#rsync -azvP -e 'ssh' ksridhar@koa02.seas.upenn.edu:~/time-series-OOD/* ./
#rsync -azvP -e 'ssh' --exclude '*.zip' ksridhar@koa02.seas.upenn.edu:/home/ramneetk/classification_on_temporal_transforms/CARLA_dataset/Vanderbilt_data ./

rsync -azvP -e 'ssh' --exclude '*.zip' ksridhar@koa02.seas.upenn.edu:/home/ramneetk/classification_on_temporal_transforms/CARLA_dataset/Vanderbilt_data/testing/out_rainy ../Vanderbilt_data/testing/