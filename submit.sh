#!/bin/sh

set -eu

cargo equip --exclude-atcoder-crates --rustfmt --check > /tmp/submit.rs
oj submit https://atcoder.jp/contests/ahc005/tasks/ahc005_a /tmp/submit.rs --yes
