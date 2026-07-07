#!/bin/bash
# CropVolare AP fallback: if the Pi can't join any known WiFi shortly after
# boot (i.e. it's in a field), broadcast its own access point instead so the
# phone GCS works with no hotspot and no cell service.
#
# At home this script does nothing: the Pi joins home WiFi within the wait
# window and exits. The AP profile ("cropvolare-ap") has autoconnect OFF, so
# this script is the ONLY thing that ever activates it - a reboot always
# returns the Pi to normal WiFi-client behavior first.
#
# Phone side, at the field:  WiFi network "CropVolare"  ->  http://10.42.0.1:8080

WAIT_TRIES=15   # x5s = 75s for a known network to connect
AP_CON=cropvolare-ap

for i in $(seq "$WAIT_TRIES"); do
    state=$(nmcli -t -f DEVICE,STATE dev | grep '^wlan0:' | cut -d: -f2)
    if [ "$state" = "connected" ]; then
        echo "wlan0 connected to a known network - AP not needed"
        exit 0
    fi
    sleep 5
done

echo "no known WiFi after $((WAIT_TRIES * 5))s - starting CropVolare access point"
nmcli connection up "$AP_CON"
