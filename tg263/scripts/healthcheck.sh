#!/bin/bash
curl -sf http://localhost:8080/healthz > /dev/null
exit $?
