python experts_test.py test.mode=render model.accelerate.bake=True --config-name replica-apartment0
python experts_test.py test.mode=render model.accelerate.bake=True --config-name replica-apartment1
python experts_test.py test.mode=render model.accelerate.bake=True --config-name replica-apartment2
python experts_test.py test.mode=render model.accelerate.bake=True --config-name replica-frl0
python experts_test.py test.mode=render model.accelerate.bake=True --config-name replica-kitchen
python experts_test.py test.mode=render model.accelerate.bake=True --config-name replica-room0
python experts_test.py test.mode=render model.accelerate.bake=True --config-name replica-room2

python -m mnh.metric -rewrite output_images/replica-apartment0/experts/color/valid/ 
python -m mnh.metric -rewrite output_images/replica-apartment1/experts/color/valid/ 
python -m mnh.metric -rewrite output_images/replica-apartment2/experts/color/valid/ 
python -m mnh.metric -rewrite output_images/replica-frl0/experts/color/valid/ 
python -m mnh.metric -rewrite output_images/replica-kitchen/experts/color/valid/
python -m mnh.metric -rewrite output_images/replica-room0/experts/color/valid/ 
python -m mnh.metric -rewrite output_images/replica-room2/experts/color/valid/ 

