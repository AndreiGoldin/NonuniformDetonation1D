for dir in `ls | grep E2`
do
cd $dir
pwd
ffmpeg -hide_banner -loglevel error -framerate 20 -start_number 0 -i image%03d.png -y -vf format=yuv420p ${dir}_video.mp4
cp *mp4 ../additional_videos
cd ..
done
