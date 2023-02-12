# datachallange

---------------------------------------
Data Challenge In The World of Sports Betting
---------------------------------------

Modellezz és tippelj egyszerre!

Vajon előre lehet jelezni egy futballmérkőzés eredményét? Ha igen, akkor ebből hogyan lehet profitálni? Ez több mint egy hagyományos data science verseny, versenyünkön egyszerre tesztelheted adatelemző és üzletielemző tudásodat. Itt nem elég, hogy a modelled minél pontosabb legyen! Ki kell találnod hozzá egy játékstratégiát, amivel tippelni fogsz a mérkőzésekre. A győztes itt nem biztos, hogy az lesz, aki legpontosabban jelzi előre a mérkőzések eredményeit, hanem az, aki a legjobb játékstratégiát találja ki a modelljéhez!

---------------------------------------
Mit kell előre jelezni? (data science verseny)
---------------------------------------

A versenyen regisztrálók kapnak egy tanító adatbázist, melyben futballmérkőzések adatai találhatók a következő formában:

Hazai csapat korábbi mérkőzéseiről aggregált adatok
Vendégcsapat korábbi mérkőzéseiről aggregált adatok
Különböző fogadóirodák mérkőzés előtti oddsainak átlaga: (i) külön a hazai győzelemre, (ii) külön a vendéggyőzelemre és (iii) külön a döntetlenre mérkőzés eredménye
A cél három különböző prediktív modell elkészítése, ahol a három célváltozó mindig kétértékűek:

modell1: hazai csapat győzelmének előrejelzése (0/1 - ahol 0, ha nem fog győzni, 1, ha győzni fog a hazai csapat)

modell2: vendégcsapat győzelmének előrejelzése (0/1 - ahol 0, ha nem fog győzni, 1, ha győzni fog a vendég csapat)

modell3: döntetlen előrejelzése (0/1 - ahol 0, ha nem döntetlen lesz, 1, ha döntetlen lesz)

Mind a három modell esetében két értéket kell kiszámolnia a modellnek:

A várható eredményt (0/1)
A várható eredmény valószínűségét (0-1 közötti érték)
A modellezésnél kizárólag a megadott adatbázisban található adatok, illetve ezen adatok alapján képzett új változók használhatók fel, ezen túl más (publikusan elérhető) adat nem. 

Az adatok forrásai publikusan elérhető adatbázisok, azonban a forrásadatok teljesen anonimizáltak lesznek, nem szerepelnek a csapatok nevei, illetve hogy melyik bajnokságok mérkőzései.  