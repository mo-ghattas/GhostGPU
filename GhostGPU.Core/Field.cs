using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace GhostGPU.Core
{
    public readonly struct V3
    {
        public readonly double X, Y, Z;
        public V3(double x, double y, double z) { X=x; Y=y; Z=z; }
        public double Length => Math.Sqrt(X*X+Y*Y+Z*Z);
        public bool Finite => Check.Finite(X) && Check.Finite(Y) && Check.Finite(Z);
        public static V3 operator +(V3 a,V3 b) => new V3(a.X+b.X,a.Y+b.Y,a.Z+b.Z);
        public static V3 operator -(V3 a,V3 b) => new V3(a.X-b.X,a.Y-b.Y,a.Z-b.Z);
        public static V3 operator *(V3 a,double b) => new V3(a.X*b,a.Y*b,a.Z*b);
        public static double Dot(V3 a,V3 b) => a.X*b.X+a.Y*b.Y+a.Z*b.Z;
        public static V3 Min(V3 a,V3 b) => new V3(Math.Min(a.X,b.X),Math.Min(a.Y,b.Y),Math.Min(a.Z,b.Z));
        public static V3 Max(V3 a,V3 b) => new V3(Math.Max(a.X,b.X),Math.Max(a.Y,b.Y),Math.Max(a.Z,b.Z));
    }

    public readonly struct Bounds
    {
        public readonly V3 Min, Max;
        public Bounds(V3 min,V3 max)
        {
            if (!min.Finite || !max.Finite || min.X>=max.X || min.Y>=max.Y || min.Z>=max.Z)
                throw new ArgumentException("Bounds must be finite and have positive extent on every axis.");
            Min=min; Max=max;
        }
        public V3 Center => Min*.5+Max*.5;
        public V3 Size => Max-Min;
        public Bounds Expand(double d) => new Bounds(Min-new V3(d,d,d),Max+new V3(d,d,d));
        public static Bounds Union(Bounds a,Bounds b) => new Bounds(V3.Min(a.Min,b.Min),V3.Max(a.Max,b.Max));
    }

    public static class Check
    {
        public static bool Finite(double d) => !double.IsNaN(d) && !double.IsInfinity(d);
        public static void Positive(double d,string name) { if(!Finite(d)||d<=0) throw new ArgumentOutOfRangeException(name,"Must be finite and positive."); }
    }

    public enum FieldOp { Sphere, Box, Union, Difference, Intersection, Gyroid }

    /// <summary>Immutable, bounded implicit field DAG. This is not an analytic Rhino Brep.</summary>
    public sealed class Field
    {
        public FieldOp Op { get; }
        public Field A { get; }
        public Field B { get; }
        public Bounds Bounds { get; }
        public int Depth { get; }
        private readonly double[] values;
        public double Parameter(int index) => values[index];
        private Field(FieldOp op,Bounds bounds,Field a,Field b,params double[] p)
        {
            Op=op; Bounds=bounds; A=a; B=b; values=(double[])p.Clone();
            Depth=1+Math.Max(a?.Depth??0,b?.Depth??0);
            if(Depth>48) throw new ArgumentException("Prototype limit: 48 nested operations.");
        }
        public static Field Sphere(V3 center,double radius)
        {
            if(!center.Finite) throw new ArgumentException("Invalid center.");
            Check.Positive(radius,nameof(radius));
            var e=new V3(radius,radius,radius);
            return new Field(FieldOp.Sphere,new Bounds(center-e,center+e),null,null,center.X,center.Y,center.Z,radius);
        }
        public static Field Box(V3 center,V3 half,V3 x,V3 y,V3 z)
        {
            if(!center.Finite||!half.Finite||!x.Finite||!y.Finite||!z.Finite) throw new ArgumentException("Invalid box.");
            Check.Positive(half.X,"X extent"); Check.Positive(half.Y,"Y extent"); Check.Positive(half.Z,"Z extent");
            if(Math.Abs(x.Length-1)>1e-8||Math.Abs(y.Length-1)>1e-8||Math.Abs(z.Length-1)>1e-8||
               Math.Abs(V3.Dot(x,y))>1e-8||Math.Abs(V3.Dot(x,z))>1e-8||Math.Abs(V3.Dot(y,z))>1e-8)
                throw new ArgumentException("Box axes must be orthonormal.");
            var e=new V3(Math.Abs(x.X)*half.X+Math.Abs(y.X)*half.Y+Math.Abs(z.X)*half.Z,
                         Math.Abs(x.Y)*half.X+Math.Abs(y.Y)*half.Y+Math.Abs(z.Y)*half.Z,
                         Math.Abs(x.Z)*half.X+Math.Abs(y.Z)*half.Y+Math.Abs(z.Z)*half.Z);
            return new Field(FieldOp.Box,new Bounds(center-e,center+e),null,null,
                center.X,center.Y,center.Z,0,x.X,x.Y,x.Z,half.X,y.X,y.Y,y.Z,half.Y,z.X,z.Y,z.Z,half.Z);
        }
        public static Field Boolean(Field a,Field b,FieldOp operation,double blend)
        {
            if(a==null||b==null) throw new ArgumentNullException("Both fields are required.");
            if(operation<FieldOp.Union||operation>FieldOp.Intersection) throw new ArgumentException("Invalid boolean operation.");
            if(!Check.Finite(blend)||blend<0) throw new ArgumentOutOfRangeException(nameof(blend));
            // Smooth subtraction and intersection are subsets of A; union expands by at most k/4.
            var bounds=operation==FieldOp.Union ? Bounds.Union(a.Bounds,b.Bounds).Expand(blend*.25) : a.Bounds;
            return new Field(operation,bounds,a,b,blend,0,0,0);
        }
        public static Field Gyroid(Field domain,double cell,double thickness)
        {
            if(domain==null) throw new ArgumentNullException(nameof(domain));
            Check.Positive(cell,nameof(cell)); Check.Positive(thickness,nameof(thickness));
            return new Field(FieldOp.Gyroid,domain.Bounds,domain,null,cell,thickness,0,0);
        }
        public double Evaluate(V3 p)
        {
            if(Op==FieldOp.Sphere) return (p-new V3(values[0],values[1],values[2])).Length-values[3];
            if(Op==FieldOp.Box)
            {
                var q=p-new V3(values[0],values[1],values[2]);
                double x=Math.Abs(V3.Dot(q,new V3(values[4],values[5],values[6])))-values[7];
                double y=Math.Abs(V3.Dot(q,new V3(values[8],values[9],values[10])))-values[11];
                double z=Math.Abs(V3.Dot(q,new V3(values[12],values[13],values[14])))-values[15];
                return new V3(Math.Max(x,0),Math.Max(y,0),Math.Max(z,0)).Length+Math.Min(Math.Max(x,Math.Max(y,z)),0);
            }
            double a=A.Evaluate(p);
            if(Op==FieldOp.Gyroid)
            {
                // Globally 1-Lipschitz field estimate, not an exact Euclidean SDF.
                double f=2*Math.PI/values[0]; var q=(p-Bounds.Center)*f;
                double g=Math.Sin(q.X)*Math.Cos(q.Y)+Math.Sin(q.Y)*Math.Cos(q.Z)+Math.Sin(q.Z)*Math.Cos(q.X);
                return Math.Max(a,Math.Abs(g)/(2*Math.Sqrt(3)*f)-values[1]*.5);
            }
            double b=B.Evaluate(p), k=values[0];
            return Op==FieldOp.Union ? SmoothMin(a,b,k) :
                Op==FieldOp.Difference ? -SmoothMin(-a,b,k) : -SmoothMin(-a,-b,k);
        }
        private static double SmoothMin(double a,double b,double k)
        {
            if(k==0) return Math.Min(a,b);
            double h=Math.Max(k-Math.Abs(a-b),0)/k;
            return Math.Min(a,b)-h*h*k*.25;
        }
        public byte[] Serialize()
        {
            using(var stream=new MemoryStream()) using(var writer=new BinaryWriter(stream))
            { writer.Write(1); int count=0; WriteNode(writer,this,ref count); return stream.ToArray(); }
        }
        private static void WriteNode(BinaryWriter w,Field f,ref int count)
        {
            if(++count>256) throw new ArgumentException("Serialized recipe exceeds 256 nodes.");
            w.Write((int)f.Op); w.Write(f.values.Length); foreach(double d in f.values) w.Write(d);
            if(f.A!=null) WriteNode(w,f.A,ref count); if(f.B!=null) WriteNode(w,f.B,ref count);
        }
        public static Field Deserialize(byte[] bytes)
        {
            if(bytes==null||bytes.Length>40000) throw new ArgumentException("Invalid recipe payload.");
            using(var s=new MemoryStream(bytes)) using(var r=new BinaryReader(s))
            {
                if(r.ReadInt32()!=1) throw new InvalidDataException("Unknown recipe version.");
                int count=0; var f=ReadNode(r,0,ref count);
                if(s.Position!=s.Length) throw new InvalidDataException("Trailing recipe data.");
                return f;
            }
        }
        private static Field ReadNode(BinaryReader r,int depth,ref int count)
        {
            if(depth>=48||++count>256) throw new InvalidDataException("Recipe limits exceeded.");
            var op=(FieldOp)r.ReadInt32(); int n=r.ReadInt32();
            int expected=op==FieldOp.Box?16:4;
            if(op<FieldOp.Sphere||op>FieldOp.Gyroid||n!=expected) throw new InvalidDataException("Invalid recipe node.");
            var p=new double[n]; for(int i=0;i<n;i++){p[i]=r.ReadDouble();if(!Check.Finite(p[i]))throw new InvalidDataException("Non-finite parameter.");}
            if(op==FieldOp.Sphere)return Sphere(new V3(p[0],p[1],p[2]),p[3]);
            if(op==FieldOp.Box)return Box(new V3(p[0],p[1],p[2]),new V3(p[7],p[11],p[15]),new V3(p[4],p[5],p[6]),new V3(p[8],p[9],p[10]),new V3(p[12],p[13],p[14]));
            var a=ReadNode(r,depth+1,ref count);
            if(op==FieldOp.Gyroid)return Gyroid(a,p[0],p[1]);
            return Boolean(a,ReadNode(r,depth+1,ref count),op,p[0]);
        }
    }

    public sealed class CompiledField
    {
        public Field Recipe { get; }
        public V3 Origin { get; }
        public Bounds LocalBounds { get; }
        public string Source { get; }
        private readonly float[] parameters;
        public float[] CopyParameters() => (float[])parameters.Clone();
        private CompiledField(Field recipe,V3 origin,string source,float[] parameters)
        { Recipe=recipe;Origin=origin;Source=source;this.parameters=parameters;LocalBounds=new Bounds(recipe.Bounds.Min-origin,recipe.Bounds.Max-origin); }
        public static CompiledField Compile(Field root)
        {
            if(root==null)throw new ArgumentNullException(nameof(root));
            var ids=new Dictionary<Field,int>(); var nodes=new List<Field>();
            Action<Field> visit=null;
            visit=f=>{if(ids.ContainsKey(f))return;if(f.A!=null)visit(f.A);if(f.B!=null)visit(f.B);if(nodes.Count>=128)throw new ArgumentException("Prototype limit: 128 unique nodes.");ids.Add(f,nodes.Count);nodes.Add(f);};
            visit(root); V3 origin=root.Bounds.Center;
            var pars=new List<float>(); var code=new StringBuilder();
            code.AppendLine("layout(std430,binding=0) readonly buffer FieldParameters { vec4 par[]; };");
            code.AppendLine("float sm(float a,float b,float k){ if(k<=0.0)return min(a,b); float h=max(k-abs(a-b),0.0)/k; return min(a,b)-h*h*k*0.25; }");
            for(int i=0;i<nodes.Count;i++)
            {
                var f=nodes[i];int slot=pars.Count/4;int n=f.Op==FieldOp.Box?16:4;
                for(int j=0;j<n;j++)
                {
                    double d=f.Parameter(j);
                    if((f.Op==FieldOp.Sphere||f.Op==FieldOp.Box)&&j<3)d-=j==0?origin.X:j==1?origin.Y:origin.Z;
                    if(!Check.Finite(d)||Math.Abs(d)>1e20||(d!=0&&Math.Abs(d)<1e-20))throw new ArgumentException("Parameter outside supported float32 range [1e-20, 1e20].");
                    pars.Add((float)d);
                }
                string p=$"par[{slot}]",a=f.A==null?null:$"f{ids[f.A]}(x)",b=f.B==null?null:$"f{ids[f.B]}(x)";
                code.Append($"float f{i}(vec3 x){{ ");
                switch(f.Op)
                {
                    case FieldOp.Sphere:code.Append($"return length(x-{p}.xyz)-{p}.w;");break;
                    case FieldOp.Box:code.Append($"vec3 q=x-{p}.xyz; vec3 d=abs(vec3(dot(q,par[{slot+1}].xyz),dot(q,par[{slot+2}].xyz),dot(q,par[{slot+3}].xyz)))-vec3(par[{slot+1}].w,par[{slot+2}].w,par[{slot+3}].w); return length(max(d,vec3(0)))+min(max(d.x,max(d.y,d.z)),0.0);");break;
                    case FieldOp.Union:code.Append($"return sm({a},{b},{p}.x);");break;
                    case FieldOp.Difference:code.Append($"return -sm(-{a},{b},{p}.x);");break;
                    case FieldOp.Intersection:code.Append($"return -sm(-{a},-{b},{p}.x);");break;
                    case FieldOp.Gyroid:
                        int centerSlot=pars.Count/4;var c=f.Bounds.Center-origin;
                        pars.AddRange(new[]{(float)c.X,(float)c.Y,(float)c.Z,0f});
                        code.Append($"float freq=6.28318530718/{p}.x; vec3 q=(x-par[{centerSlot}].xyz)*freq; float g=sin(q.x)*cos(q.y)+sin(q.y)*cos(q.z)+sin(q.z)*cos(q.x); return max({a},abs(g)/(3.46410161514*freq)-{p}.y*0.5);");break;
                }
                code.AppendLine(" }");
            }
            code.AppendLine($"float field(vec3 x){{return f{ids[root]}(x);}}");
            return new CompiledField(root,origin,code.ToString(),pars.ToArray());
        }
    }
}
