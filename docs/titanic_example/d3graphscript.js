//Constants for the SVG
var width = 1500;
var height = 800;

//Set up the colour scale
var color = d3.scale.category20();

var force = d3.layout.force()
  .charge(-500)
  
  .size([width, height]);


//Append a SVG to the body of the html page. Assign this SVG as an object to svg
var svg = d3.select("body").append("svg")
  .attr("width", width)
  .attr("height", height)
//.on("dblclick", threshold); // EXPLODE ALL CONNECTED POINTS

//Read the data from the d3graph element 
var d3graph = document.getElementById('d3graph').innerHTML;
graph = JSON.parse(d3graph);
graphRec = JSON.parse(JSON.stringify(graph));

//Creates the graph data structure out of the json data
force.nodes(graph.nodes)
  .links(graph.links)
  .start();

//Create all the line svgs but without locations yet
var link = svg.selectAll(".link")
  .data(graph.links)
  .enter().append("line")
  .attr("class", "link")
  ////.style('marker-start',  'url(#suit)') // ARROWS IN EDGES
  //.style('marker-end',  'url(#suit)') // ARROWS IN EDGES
  .style("stroke-width", function(d) {return d.edge_width;}) // LINK-WIDTH
  ;
//  .style("stroke-width", 1); // WIDTH OF THE LINKS

//Do the same with the circles for the nodes
var node = svg.selectAll(".node")
  .data(graph.nodes)
  .enter().append("g")
  .attr("class", "node")
  .call(force.drag)
  .on('dblclick', connectedNodes); //Highliht ON/OFF

node.append("circle")
  .attr("r", function(d) { return d.node_size; })					// NODE SIZE
  .style("fill", function(d) {return d.node_color;})				// NODE-COLOR
  .style("stroke-width", function(d) {return d.node_size_edge;})	// NODE-EDGE-SIZE
  .style("stroke", function(d) {return d.node_color_edge;})			// NODE-COLOR-EDGE
//  .style("stroke", '#000')										// NODE-EDGE-COLOR (all black)

// Text in nodes
node.append("text")
  .attr("dx", 10)
  .attr("dy", ".35em")
  .text(function(d) {return d.node_name}) // NODE-TEXT
//  .style("stroke", "gray");

node.append("title")
	.text(function(d) { return "Node: " + d.id + "\n" + "node_name: " + d.node_name ;}); // HOVEROVER NODE TEXT

//Now we are giving the SVGs co-ordinates - the force layout is generating the co-ordinates which this code is using to update the attributes of the SVG elements
force.on("tick", function() {
  link.attr("x1", function(d) {
      return d.source.x;
    })
    .attr("y1", function(d) {
      return d.source.y;
    })
    .attr("x2", function(d) {
      return d.target.x;
    })
    .attr("y2", function(d) {
      return d.target.y;
    });
  d3.selectAll("circle").attr("cx", function(d) {
      return d.x;
    })
    .attr("cy", function(d) {
      return d.y;
    });
  d3.selectAll("text").attr("x", function(d) {
      return d.x;
    })
    .attr("y", function(d) {
      return d.y;
    });

  node.each(collide(0.1)); //COLLISION DETECTION. High means a big fight to get untouchable nodes (default=0.5)

});

// --------- Directed lines -----------
svg.append("defs").selectAll("marker")
    .data(["suit", "licensing", "resolved"])
  .enter().append("marker")
    .attr("id", function(d) { return d; })
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 25)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("path")
    .attr("d", "M0,-5L10,0L0,5 L10,0 L0, -5")
    .style("stroke", "#4679BD")
    .style("opacity", "0.6");

// --------- End directed lines -----------

//---Insert-------


// collision detection

var padding = 1, // separation between circles
  radius = 8;

function collide(alpha) {
  var quadtree = d3.geom.quadtree(graph.nodes);
  return function(d) {
    var rb = 2 * radius + padding,
      nx1 = d.x - rb,
      nx2 = d.x + rb,
      ny1 = d.y - rb,
      ny2 = d.y + rb;
    quadtree.visit(function(quad, x1, y1, x2, y2) {
      if (quad.point && (quad.point !== d)) {
        var x = d.x - quad.point.x,
          y = d.y - quad.point.y,
          l = Math.sqrt(x * x + y * y);
        if (l < rb) {
          l = (l - rb) / l * alpha;
          d.x -= x *= l;
          d.y -= y *= l;
          quad.point.x += x;
          quad.point.y += y;
        }
      }
      return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
    });
  };
}
// collision detection end


//Toggle stores whether the highlighting is on **********************
var toggle = 0;
//Create an array logging what is connected to what
var linkedByIndex = {};
for (i = 0; i < graph.nodes.length; i++) {
  linkedByIndex[i + "," + i] = 1;
};
graph.links.forEach(function(d) {
  linkedByIndex[d.source.index + "," + d.target.index] = 1;
});
//This function looks up whether a pair are neighbours
function neighboring(a, b) {
  return linkedByIndex[a.index + "," + b.index];
}

function connectedNodes() {
  if (toggle == 0) {
    //Reduce the opacity of all but the neighbouring nodes
    d = d3.select(this).node().__data__;
    node.style("opacity", function(o) {
      return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
    });
    link.style("opacity", function(o) {
      return d.index == o.source.index | d.index == o.target.index ? 1 : 0.1;
    });
    //Reduce the op
    toggle = 1;
  } else {
    //Put them back to opacity=1
    node.style("opacity", 1);
    link.style("opacity", 1);

    toggle = 0;
  }
}
//*************************************************************


//adjust threshold
function threshold(thresh) {
  graph.links.splice(0, graph.links.length);

  for (var i = 0; i < graphRec.links.length; i++) {
    if (graphRec.links[i].edge_weight > thresh) {
      graph.links.push(graphRec.links[i]);
    }
  }
  restart();
}

//Restart the visualisation after any node and link changes
function restart() {

  link = link.data(graph.links);
  link.exit().remove();
  link.enter().insert("line", ".node").attr("class", "link");
  link.style("stroke-width", function(d) {return d.edge_width;}); // WIDTH OF THE LINKS AFTER BREAKING WITH SLIDER
  ////link.style('marker-start','url(#suit)') // ARROWS IN EDGES
  //link.style('marker-end','url(#suit)') // ARROWS IN EDGES
  node = node.data(graph.nodes);
  node.enter().insert("circle", ".cursor").attr("class", "node").attr("r", 5).call(force.drag);
  force.start();
}


//---End Insert---
