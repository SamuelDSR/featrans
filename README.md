# featrans
A light, and pure python package that aims to create a *"port"* between pyspark and python.
It allows you write any feature transforms using the **wrapper class** located in the **spark** subpackage on standard spark DataFrame,
save the entire feature transform pipeline in a file, and restore the pipeline in pure python program, 
saving a lot of time
when you train a model in **pyspark** and want to use this model in a **pure** python program, for example, online production.

More to add
