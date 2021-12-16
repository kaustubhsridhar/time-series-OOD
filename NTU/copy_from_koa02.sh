#rsync -azvP -e 'ssh' ksridhar@koa02.seas.upenn.edu:~/time-series-OOD/* ./
#rsync -azvP -e 'ssh' --exclude '*.zip' ksridhar@koa02.seas.upenn.edu:/home/ramneetk/classification_on_temporal_transforms/CARLA_dataset/Vanderbilt_data ./

#rsync -azvP -e 'ssh' --exclude '*.zip' ksridhar@koa02.seas.upenn.edu:/home/ramneetk/classification_on_temporal_transforms/CARLA_dataset/Vanderbilt_data/testing/out_rainy ../Vanderbilt_data/testing/

rsync -azvP -e 'ssh' --exclude '*.zip' ksridhar@koa02.seas.upenn.edu:/home/ramneetk/classification_on_temporal_transforms/drift_dataset/temp_in/training/ ../drift_data/training/
rsync -azvP -e 'ssh' --exclude '*.zip' ksridhar@koa02.seas.upenn.edu:/home/ramneetk/classification_on_temporal_transforms/drift_dataset/temp_in/testing/ ../drift_data/testing/in/
rsync -azvP -e 'ssh' --exclude '*.zip' ksridhar@koa02.seas.upenn.edu:/home/ramneetk/classification_on_temporal_transforms/drift_dataset/out/ ../drift_data/testing/out/

