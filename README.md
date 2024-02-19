## Assignment 2


launch.py script location:

```bash
/home/cs744/miniconda3/lib/python3.11/site-packages/torch/distributed/launch.py
```
### Node Reference
A primary node is running with 3 distributed docker containers according to:

- "node0":"172.18.0.2:60000"
- "node1":"172.18.0.3:60001"
- "node2":"172.18.0.4:60002"
- "node3":"172.18.0.5:60003"

### Homework 2: Remaining Tasks

### Code [AT]
from node 0 on 8d5b5e0ed9c2:
- `./home/cs744/hw2/part1_main.py`
- `./home/cs744/hw2/part2a_main.py`
    - add docstrings to functions like `average_gradients()`
- `./home/cs744/hw2/part2b_main.py`
    - add docstrings to functions like `average_gradients()`
- `./home/cs744/hw2/part3_main.py`

For all code:
- Docstring for `argparse`
- Ensure they are runnable through command line as: `python main.py --master-ip $ip_address$ --num-nodes 4 --rank $rank$`
- Comment all relevant code in portion left to assignment (small amount)

### README.md file [AT]
- Create a final project directory `README.md`


### Network traffic logs

`docker run --privileged <your_docker_image> tcpdump -i eth0 -w /path/to/output.pcap`

### Final Report
[Team to work on this during the week]
- Template
- Data

