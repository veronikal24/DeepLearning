for i in {11..14}; do
  wget http://aisdata.ais.dk/aisdk-2025-10-$(printf "%02d" "$i").zip
  unzip aisdk-2025-10-$(printf "%02d" "$i").zip
  rm aisdk-2025-10-$(printf "%02d" "$i").zip
  python dataloader.py aisdk-2025-10-$(printf "%02d" "$i").csv
  rm aisdk-2025-10-$(printf "%02d" "$i").csv
  echo "Processed file $i"
done