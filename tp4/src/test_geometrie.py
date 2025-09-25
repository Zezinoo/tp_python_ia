from geometrie import Point , Scene , Cercle  , Rectangle , Transformation2D , Vecteur2D
from matplotlib import pyplot as plt



def test_vec():

    p = Point(2,1)
    print(p)
    v = p.vers_vecteur()
    normed = v.normaliser()
    gauche = v.perpendiculaire_gauche()
    origin = [0,0]
    vecs = [v , normed , gauche]
    colors = ["r" , "g" , "b"]


    fig , ax = plt.subplots(1,1,figsize=(10,4))
    for vec, c in zip(vecs, colors):
        ax.quiver(
            0, 0,              # start at origin
            vec.x, vec.y,      # vector components
            angles='xy',
            scale_units='xy',
            scale=1,
            color=c
        )
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')    
    plt.show()


def main():
    #test_vec()
    p = Point(1,1)
    print(p)
    c = Cercle(p , rayon= 2)
    r = Rectangle( p , largeur= 3 , hauteur= 4)
    s = Scene()
    s.ajouter(c)
    s.ajouter(r)
    s.translater_tout(10 , 2)
    s.tracer()
    pass

if __name__ == "__main__":
    main()