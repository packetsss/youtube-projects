# pool-game
### Check out my [YouTube tutorial](https://www.youtube.com/watch?v=a9yDqKloXf4&list=PLe2NghaCZ6ovKhmCeIUYCykno4-el_Efj)!

<img src="https://cdn.discordapp.com/attachments/763819251249184789/885387030333358101/Animation.gif" width=450> <img src="https://github.com/packetsss/youtube-projects/blob/main/pool-game/demo/Animation.gif?raw=true" width=450>


(FPS in the gif is splited by 10 instances, so it should be around 110)

## Quick Start
```
# install packages
pip install -r requirements.txt

# run pool simulator
python pool.py

# run genetic algorithm solver
python genetic.py
```

## Advantages of using this environment
- Fast: About 250 fps using vector observation, 120 fps using image observation
- Accurate: Every collision is handeled correctly and the table size is very precise
- Customizable: You can tweak with configs to get your favourite table, pretty easy and I've commented basically on everything
