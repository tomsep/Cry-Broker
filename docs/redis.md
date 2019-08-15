# REDIS CONFIGURATION
Build the docker container

	cd ./redis
	sudo docker build --tag=redis-server .

Start it

	sudo docker run -p 6379:6379 redis-server

To run multiple redis instances you have to change the port mapping to avoid conflicts, e.g. 7000:6379 for the second instance.

Remember to configure redis password in ./redis/redis.conf.