#!/bin/bash
cd /home/vtg/Coding-Agent/generations/CodingAgent
./venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8000
