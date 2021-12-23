import perceptron as p
import training as t
import time
             

perc = p.Perceptron(learning_rate=0.001)             
size = (t.screen_width,t.screen_height)
#inputs = (1,0.5)    
        
#px = self.map_range(self.x, -1, 1, 0, screen_width)
#py = self.map_range(self.y, -1, 1, screen_height, 0)
number_of_points = 1000
points = []
# Vyplnit pole jednotlivými objekty         
for i in range(number_of_points):
    points.append(t.Point(size=size))

def train():
    for point in points:
        target = point.label
        cords = (point.x,point.y, point.bias)
        perc.train(cords,target)
        guess = perc.guess(cords)
        if guess == target:
            t.pygame.draw.circle(t.screen, t.GREEN, (point.px,point.py),t.DOT_SIZE/2)
        else:
            t.pygame.draw.circle(t.screen, t.RED, (point.px,point.py),t.DOT_SIZE/2)
        #p3 = t.Point(-1, perc.guess_Y(-1))
        #p4 = t.Point(1, perc.guess_Y(1))
        #t.pygame.draw.line(t.screen, t.BLUE, (p3.px,p3.py), (p4.px,p4.py), 1)
        #t.pygame.display.update()
        #t.pygame.draw.line(t.screen, t.WHITE, (p3.px,p3.py), (p4.px,p4.py), 1)
    print("Done training")

def draw(points):
    p1 = t.Point(-1, t.f(-1))
    p2 = t.Point(1, t.f(1))
    t.pygame.draw.line(t.screen, t.BLACK, (p1.px,p1.py), (p2.px,p2.py), 2)

    for point in points:
        point.show()
        target = point.label
        cords = (point.x,point.y, point.bias)
        guess = perc.guess(cords)
        if guess == target:
            t.pygame.draw.circle(t.screen, t.GREEN, (point.px,point.py),t.DOT_SIZE/2)
        else:
            t.pygame.draw.circle(t.screen, t.RED, (point.px,point.py),t.DOT_SIZE/2)
        #t.pygame.display.update()
    
    print("Done drawing")

draw(points)
generace = 0
while True:
    if t.pygame.key.get_pressed()[t.pygame.K_SPACE]:
        print("Generace:", generace)
        generace += 1
        train()
        time.sleep(1)

    event_list = t.pygame.event.get()
    for event in event_list:
        if event.type == t.pygame.QUIT:
            t.pygame.quit()
            quit()
        
    t.pygame.display.update()
