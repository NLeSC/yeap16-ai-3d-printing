#!/bin/sh

app=$(which Slicer)
#$app --no-main-window --no-splash --launcher-dump-environment
OPTS="--exit-after-startup \
    --no-main-window --no-splash --disable-tooltips \
    --disable-message-handlers \
    --disable-settings --ignore-slicerrc \
    --exit-after-startup "
#$app $OPTS -c $1
$app $OPTS --python-script $1
